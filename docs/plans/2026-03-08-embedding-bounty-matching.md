# Embedding-Based Bounty Matching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace LLM summarization in bounty_search with embedding similarity via embed-server, keeping LLM as fallback.

**Architecture:** Call embed-server (Rust ONNX, multilingual-e5-large, 1024-dim) directly from go-job to embed bounty texts and user query. Compute cosine similarity to rank/filter bounties. Extract skills via GitHub API labels + keyword dictionary. Fall back to existing LLM pipeline if embed-server is unavailable.

**Tech Stack:** Go, embed-server (OpenAI-compatible `/v1/embeddings`), multilingual-e5-large (1024-dim), GitHub API

---

### Task 1: Embed Client

**Files:**
- Create: `internal/engine/jobs/embed.go`
- Create: `internal/engine/jobs/embed_test.go`
- Modify: `internal/engine/config.go:18-52` (add EmbedURL field)
- Modify: `main.go:67-89` (read EMBED_URL env)

**Context:** embed-server runs at `http://embed-server:8082` with OpenAI-compatible API:
```
POST /v1/embeddings
{"input": ["text1", "text2"]}
→ {"data": [{"embedding": [f32...], "index": 0}, ...]}
```

**Step 1: Add EmbedURL to Config**

In `internal/engine/config.go`, add field to Config struct after `MemDBServiceSecret`:

```go
EmbedURL string // EMBED_URL for direct embedding server
```

In `main.go`, in `initEngine()`, add after line 84 (`MemDBServiceSecret`):

```go
EmbedURL: env.Str("EMBED_URL", ""),
```

**Step 2: Write the failing test**

Create `internal/engine/jobs/embed_test.go`:

```go
package jobs

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEmbedTexts(t *testing.T) {
	t.Parallel()

	// Mock embed-server returning 2-dim vectors for simplicity.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		var req embedRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		resp := embedResponse{Object: "list", Model: "test"}
		for i := range req.Input {
			resp.Data = append(resp.Data, embedData{
				Object:    "embedding",
				Embedding: []float32{0.5, 0.5}, // dummy vector
				Index:     i,
			})
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	client := NewEmbedClient(srv.URL)
	vecs, err := client.EmbedTexts(context.Background(), []string{"hello", "world"})
	if err != nil {
		t.Fatalf("EmbedTexts failed: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(vecs))
	}
	if len(vecs[0]) != 2 {
		t.Errorf("expected 2-dim vector, got %d", len(vecs[0]))
	}
}

func TestEmbedTexts_EmptyInput(t *testing.T) {
	t.Parallel()

	client := NewEmbedClient("http://localhost:1") // won't be called
	vecs, err := client.EmbedTexts(context.Background(), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(vecs) != 0 {
		t.Errorf("expected 0 vectors, got %d", len(vecs))
	}
}

func TestCosineSimilarity(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		a, b []float32
		want float32
	}{
		{"identical", []float32{1, 0}, []float32{1, 0}, 1.0},
		{"orthogonal", []float32{1, 0}, []float32{0, 1}, 0.0},
		{"opposite", []float32{1, 0}, []float32{-1, 0}, -1.0},
		{"similar", []float32{0.8, 0.6}, []float32{0.6, 0.8}, 0.96},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := CosineSimilarity(tt.a, tt.b)
			if math.Abs(float64(got-tt.want)) > 0.01 {
				t.Errorf("CosineSimilarity() = %f, want %f", got, tt.want)
			}
		})
	}
}

func TestCosineSimilarity_DifferentLengths(t *testing.T) {
	t.Parallel()
	got := CosineSimilarity([]float32{1, 0}, []float32{1})
	if got != 0 {
		t.Errorf("expected 0 for mismatched lengths, got %f", got)
	}
}
```

**Step 3: Run test to verify it fails**

Run: `cd /home/krolik/src/go-job && go test -buildvcs=false ./internal/engine/jobs/ -run "TestEmbedTexts|TestCosineSimilarity" -v`
Expected: FAIL — `NewEmbedClient` undefined

**Step 4: Write minimal implementation**

Create `internal/engine/jobs/embed.go`:

```go
package jobs

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"time"

	"github.com/anatolykoptev/go_job/internal/engine"
)

// EmbedClient calls an OpenAI-compatible embedding server.
type EmbedClient struct {
	baseURL string
	http    *http.Client
}

// NewEmbedClient creates an embed client for the given base URL.
func NewEmbedClient(baseURL string) *EmbedClient {
	return &EmbedClient{
		baseURL: baseURL,
		http:    &http.Client{Timeout: 30 * time.Second},
	}
}

type embedRequest struct {
	Input []string `json:"input"`
	Model string   `json:"model,omitempty"`
}

type embedResponse struct {
	Object string      `json:"object"`
	Data   []embedData `json:"data"`
	Model  string      `json:"model"`
}

type embedData struct {
	Object    string    `json:"object"`
	Embedding []float32 `json:"embedding"`
	Index     int       `json:"index"`
}

// EmbedTexts sends texts to the embedding server and returns vectors.
func (c *EmbedClient) EmbedTexts(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body, err := json.Marshal(embedRequest{Input: texts})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embed-server returned status %d", resp.StatusCode)
	}

	var result embedResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	// Sort by index to ensure correct order.
	vecs := make([][]float32, len(result.Data))
	for _, d := range result.Data {
		if d.Index < len(vecs) {
			vecs[d.Index] = d.Embedding
		}
	}
	return vecs, nil
}

// Healthy checks if embed-server is reachable.
func (c *EmbedClient) Healthy(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/health", nil)
	if err != nil {
		return false
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}

// CosineSimilarity computes cosine similarity between two vectors.
func CosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return float32(dot / denom)
}

// Package-level embed client, set by SetEmbedClient.
var embedClient *EmbedClient

// SetEmbedClient sets the package-level embed client.
func SetEmbedClient(c *EmbedClient) { embedClient = c }

// GetEmbedClient returns the package-level embed client (nil if not configured).
func GetEmbedClient() *EmbedClient { return embedClient }
```

**Step 5: Run test to verify it passes**

Run: `cd /home/krolik/src/go-job && go test -buildvcs=false ./internal/engine/jobs/ -run "TestEmbedTexts|TestCosineSimilarity" -v`
Expected: PASS (all 4 tests)

**Step 6: Wire up in main.go**

In `main.go`, after memdb init block (line 136), add:

```go
// Embed client (for embedding-based bounty matching)
if c.EmbedURL != "" {
	jobs.SetEmbedClient(jobs.NewEmbedClient(c.EmbedURL))
	slog.Info("embed client initialized", slog.String("url", c.EmbedURL))
}
```

**Step 7: Add EMBED_URL to docker compose**

In `/home/krolik/deploy/krolik-server/compose/apps.yml`, in go-job environment section, add:

```yaml
- EMBED_URL=http://embed-server:8082
```

**Step 8: Build and verify**

Run: `cd /home/krolik/src/go-job && go build -buildvcs=false ./...`
Expected: clean build

**Step 9: Commit**

```bash
cd /home/krolik/src/go-job
git add internal/engine/jobs/embed.go internal/engine/jobs/embed_test.go internal/engine/config.go main.go
git commit -m "feat: add embed client for embedding-based bounty matching"
```

---

### Task 2: Skills Extraction

**Files:**
- Create: `internal/engine/jobs/skills.go`
- Create: `internal/engine/jobs/skills_test.go`

**Context:** Extract skills from two sources:
1. GitHub API: issue labels + repo primary language
2. Keyword dictionary: regex match against title + issue body

**Step 1: Write the failing test**

Create `internal/engine/jobs/skills_test.go`:

```go
package jobs

import (
	"testing"
)

func TestExtractSkillsFromText(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name string
		text string
		want []string // at least these skills should be found
	}{
		{
			name: "golang and rust",
			text: "We need a Go developer with Rust experience to build a CLI tool",
			want: []string{"Go", "Rust", "CLI"},
		},
		{
			name: "web stack",
			text: "React frontend with TypeScript, Node.js backend, PostgreSQL database",
			want: []string{"React", "TypeScript", "Node.js", "PostgreSQL"},
		},
		{
			name: "scala zio",
			text: "Schema Migration System for ZIO Schema 2 using Scala macros",
			want: []string{"Scala", "ZIO"},
		},
		{
			name: "MCP and AI",
			text: "Incorporate MCP Server into the CLI for AI agent integration",
			want: []string{"MCP", "AI", "CLI"},
		},
		{
			name: "empty text",
			text: "",
			want: nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractSkillsFromText(tt.text)
			for _, w := range tt.want {
				found := false
				for _, g := range got {
					if g == w {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("ExtractSkillsFromText() missing %q, got %v", w, got)
				}
			}
		})
	}
}

func TestMergeSkills(t *testing.T) {
	t.Parallel()

	got := MergeSkills(
		[]string{"Go", "Rust", "go"},
		[]string{"Python", "Go", "rust"},
	)
	// Should deduplicate case-insensitively, keep first casing.
	if len(got) < 3 {
		t.Errorf("expected at least 3 unique skills, got %v", got)
	}
	// Should be sorted.
	for i := 1; i < len(got); i++ {
		if got[i] < got[i-1] {
			t.Errorf("not sorted: %v", got)
			break
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /home/krolik/src/go-job && go test -buildvcs=false ./internal/engine/jobs/ -run "TestExtractSkills|TestMergeSkills" -v`
Expected: FAIL — `ExtractSkillsFromText` undefined

**Step 3: Write minimal implementation**

Create `internal/engine/jobs/skills.go`:

```go
package jobs

import (
	"regexp"
	"sort"
	"strings"
)

// skillPatterns maps canonical skill names to their regex patterns.
// Patterns use word boundaries (\b) for accurate matching.
var skillPatterns = []struct {
	name    string
	pattern *regexp.Regexp
}{
	// Languages
	{"Go", regexp.MustCompile(`\b(?:Go|Golang|golang)\b`)},
	{"Rust", regexp.MustCompile(`\bRust\b`)},
	{"Python", regexp.MustCompile(`\bPython\b`)},
	{"TypeScript", regexp.MustCompile(`\bTypeScript\b`)},
	{"JavaScript", regexp.MustCompile(`\bJavaScript\b`)},
	{"Scala", regexp.MustCompile(`\bScala\b`)},
	{"Java", regexp.MustCompile(`\bJava\b`)},
	{"C\u002B\u002B", regexp.MustCompile(`\bC\+\+\b`)},
	{"C#", regexp.MustCompile(`\bC#\b`)},
	{"Ruby", regexp.MustCompile(`\bRuby\b`)},
	{"PHP", regexp.MustCompile(`\bPHP\b`)},
	{"Swift", regexp.MustCompile(`\bSwift\b`)},
	{"Kotlin", regexp.MustCompile(`\bKotlin\b`)},
	{"Elixir", regexp.MustCompile(`\bElixir\b`)},
	{"Haskell", regexp.MustCompile(`\bHaskell\b`)},
	{"Zig", regexp.MustCompile(`\bZig\b`)},
	// Frameworks & Libraries
	{"React", regexp.MustCompile(`\bReact\b`)},
	{"Vue", regexp.MustCompile(`\bVue\.?js?\b`)},
	{"Angular", regexp.MustCompile(`\bAngular\b`)},
	{"Next.js", regexp.MustCompile(`\bNext\.?js\b`)},
	{"Node.js", regexp.MustCompile(`\bNode\.?js\b`)},
	{"Django", regexp.MustCompile(`\bDjango\b`)},
	{"Flask", regexp.MustCompile(`\bFlask\b`)},
	{"FastAPI", regexp.MustCompile(`\bFastAPI\b`)},
	{"Spring", regexp.MustCompile(`\bSpring\b`)},
	{"Rails", regexp.MustCompile(`\bRails\b`)},
	{"ZIO", regexp.MustCompile(`\bZIO\b`)},
	// Databases
	{"PostgreSQL", regexp.MustCompile(`\b(?:PostgreSQL|Postgres)\b`)},
	{"MySQL", regexp.MustCompile(`\bMySQL\b`)},
	{"MongoDB", regexp.MustCompile(`\bMongo(?:DB)?\b`)},
	{"Redis", regexp.MustCompile(`\bRedis\b`)},
	{"SQLite", regexp.MustCompile(`\bSQLite\b`)},
	// Infrastructure & Tools
	{"Docker", regexp.MustCompile(`\bDocker\b`)},
	{"Kubernetes", regexp.MustCompile(`\b(?:Kubernetes|K8s)\b`)},
	{"AWS", regexp.MustCompile(`\bAWS\b`)},
	{"GCP", regexp.MustCompile(`\bGCP\b`)},
	{"Azure", regexp.MustCompile(`\bAzure\b`)},
	{"Terraform", regexp.MustCompile(`\bTerraform\b`)},
	{"CI/CD", regexp.MustCompile(`\bCI/?CD\b`)},
	{"GraphQL", regexp.MustCompile(`\bGraphQL\b`)},
	{"gRPC", regexp.MustCompile(`\bgRPC\b`)},
	{"REST", regexp.MustCompile(`\bREST(?:ful)?\b`)},
	// AI & ML
	{"AI", regexp.MustCompile(`\bAI\b`)},
	{"ML", regexp.MustCompile(`\bML\b`)},
	{"LLM", regexp.MustCompile(`\bLLM\b`)},
	{"MCP", regexp.MustCompile(`\bMCP\b`)},
	{"NLP", regexp.MustCompile(`\bNLP\b`)},
	// Concepts
	{"CLI", regexp.MustCompile(`\bCLI\b`)},
	{"API", regexp.MustCompile(`\bAPI\b`)},
	{"WebSocket", regexp.MustCompile(`\bWebSocket\b`)},
	{"WASM", regexp.MustCompile(`\b(?:WASM|WebAssembly)\b`)},
	{"NIO", regexp.MustCompile(`\bNIO\b`)},
	{"XML", regexp.MustCompile(`\bXML\b`)},
	{"JSON", regexp.MustCompile(`\bJSON\b`)},
}

// ExtractSkillsFromText finds known technology keywords in text.
func ExtractSkillsFromText(text string) []string {
	if text == "" {
		return nil
	}
	var skills []string
	for _, sp := range skillPatterns {
		if sp.pattern.MatchString(text) {
			skills = append(skills, sp.name)
		}
	}
	sort.Strings(skills)
	return skills
}

// MergeSkills merges multiple skill slices, deduplicating case-insensitively.
// Keeps the first casing encountered. Returns sorted.
func MergeSkills(slices ...[]string) []string {
	seen := make(map[string]string) // lowercase → original
	for _, s := range slices {
		for _, skill := range s {
			lower := strings.ToLower(skill)
			if _, ok := seen[lower]; !ok {
				seen[lower] = skill
			}
		}
	}
	result := make([]string, 0, len(seen))
	for _, v := range seen {
		result = append(result, v)
	}
	sort.Strings(result)
	return result
}
```

**Step 4: Run test to verify it passes**

Run: `cd /home/krolik/src/go-job && go test -buildvcs=false ./internal/engine/jobs/ -run "TestExtractSkills|TestMergeSkills" -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/krolik/src/go-job
git add internal/engine/jobs/skills.go internal/engine/jobs/skills_test.go
git commit -m "feat: add keyword-based skills extraction for bounties"
```

---

### Task 3: Embedding Pipeline in tool_bounty.go

**Files:**
- Modify: `internal/jobserver/tool_bounty.go` (replace LLM with embedding, keep LLM as fallback)
- Modify: `internal/engine/jobs/algora.go` (add ParseAmountCents helper)

**Context:** The current `tool_bounty.go` calls `SummarizeBountyResults` (LLM). Replace primary path with:
1. Embed bounty texts (title + issue content)
2. If query non-empty: embed query, compute cosine similarity, sort by score, filter < 0.3
3. If query empty: sort by amount descending
4. Extract skills from issue content via keyword matching
5. If embed-server unavailable: fall back to LLM pipeline

**Step 1: Add ParseAmountCents to algora.go**

In `internal/engine/jobs/algora.go`, add after `extractTitleFromBlock`:

```go
// ParseAmountCents parses "$4,000" → 400000 (cents) for sorting.
func ParseAmountCents(amount string) int {
	s := strings.ReplaceAll(amount, "$", "")
	s = strings.ReplaceAll(s, ",", "")
	s = strings.TrimSpace(s)
	n, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return n * 100
}
```

**Step 2: Add test for ParseAmountCents**

In `internal/engine/jobs/algora_test.go`, add:

```go
func TestParseAmountCents(t *testing.T) {
	t.Parallel()

	tests := []struct {
		input string
		want  int
	}{
		{"$4,000", 400000},
		{"$500", 50000},
		{"$3,500", 350000},
		{"$100", 10000},
		{"", 0},
		{"free", 0},
	}
	for _, tt := range tests {
		got := ParseAmountCents(tt.input)
		if got != tt.want {
			t.Errorf("ParseAmountCents(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}
```

**Step 3: Run test**

Run: `cd /home/krolik/src/go-job && go test -buildvcs=false ./internal/engine/jobs/ -run "TestParseAmountCents" -v`
Expected: PASS

**Step 4: Rewrite tool_bounty.go**

Replace `internal/jobserver/tool_bounty.go` with:

```go
package jobserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"sort"

	"github.com/anatolykoptev/go_job/internal/engine"
	"github.com/anatolykoptev/go_job/internal/engine/jobs"
	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func registerBountySearch(server *mcp.Server) {
	mcp.AddTool(server, &mcp.Tool{
		Name:        "bounty_search",
		Description: "Search for open-source bounties on Algora.io. Returns paid GitHub issues with bounty amounts. Filter by technology or keyword. Great for finding paid open-source contribution opportunities.",
		Annotations: &mcp.ToolAnnotations{ReadOnlyHint: true},
	}, func(ctx context.Context, req *mcp.CallToolRequest, input engine.BountySearchInput) (*mcp.CallToolResult, engine.SmartSearchOutput, error) {
		// Fetch bounties from Algora (scrape result is cached internally).
		bounties, err := jobs.SearchAlgora(ctx, 30)
		if err != nil {
			slog.Warn("bounty_search: algora error", slog.Any("error", err))
		}

		if len(bounties) == 0 {
			if err != nil {
				return nil, engine.SmartSearchOutput{}, fmt.Errorf("algora fetch failed: %w", err)
			}
			return bountyResult(engine.BountySearchOutput{Query: input.Query, Summary: "No bounties found."})
		}

		// Filter out closed/completed GitHub issues (always fresh, not cached).
		bounties = jobs.FilterOpenBounties(ctx, bounties)
		if len(bounties) == 0 {
			return bountyResult(engine.BountySearchOutput{Query: input.Query, Summary: "All bounties found are already completed."})
		}

		// Fetch GitHub issue contents for context.
		searxResults := jobs.BountiesToSearxngResults(bounties)
		apiURLs := make(map[string]bool)
		contents := engine.FetchContentsParallel(ctx, searxResults, apiURLs)

		// Extract skills from issue content via keyword matching.
		for i := range bounties {
			issueText := contents[bounties[i].URL]
			textSkills := jobs.ExtractSkillsFromText(bounties[i].Title + " " + issueText)
			bounties[i].Skills = jobs.MergeSkills(bounties[i].Skills, textSkills)
		}

		// Try embedding pipeline first, fallback to LLM.
		embedClient := jobs.GetEmbedClient()
		if embedClient != nil && embedClient.Healthy(ctx) {
			result, err := bountyEmbedPipeline(ctx, embedClient, input.Query, bounties, contents)
			if err != nil {
				slog.Warn("bounty_search: embed pipeline failed, falling back to LLM", slog.Any("error", err))
			} else {
				return bountyResult(result)
			}
		}

		// Fallback: LLM pipeline.
		return bountyLLMFallback(ctx, input, bounties, searxResults, contents)
	})
}

// bountyEmbedPipeline uses embedding similarity for query matching and sorting.
func bountyEmbedPipeline(ctx context.Context, client *jobs.EmbedClient, query string, bounties []engine.BountyListing, contents map[string]string) (engine.BountySearchOutput, error) {
	if query == "" {
		// No query — sort by amount descending.
		sort.Slice(bounties, func(i, j int) bool {
			return jobs.ParseAmountCents(bounties[i].Amount) > jobs.ParseAmountCents(bounties[j].Amount)
		})
		return engine.BountySearchOutput{
			Query:    query,
			Bounties: bounties,
		}, nil
	}

	// Build texts for embedding: title + issue content.
	texts := make([]string, len(bounties))
	for i, b := range bounties {
		text := b.Title
		if c, ok := contents[b.URL]; ok && c != "" {
			text += "\n" + c
		}
		// Truncate to ~500 chars for embedding efficiency.
		if len(text) > 500 {
			text = text[:500]
		}
		texts[i] = text
	}

	// Embed query + bounty texts in single batch.
	allTexts := append([]string{query}, texts...)
	vecs, err := client.EmbedTexts(ctx, allTexts)
	if err != nil {
		return engine.BountySearchOutput{}, fmt.Errorf("embedding failed: %w", err)
	}
	if len(vecs) != len(allTexts) {
		return engine.BountySearchOutput{}, fmt.Errorf("embed returned %d vectors, expected %d", len(vecs), len(allTexts))
	}

	queryVec := vecs[0]
	bountyVecs := vecs[1:]

	// Compute similarity and filter.
	type scored struct {
		bounty engine.BountyListing
		score  float32
	}
	var results []scored
	for i, bv := range bountyVecs {
		sim := jobs.CosineSimilarity(queryVec, bv)
		if sim >= 0.3 {
			results = append(results, scored{bounty: bounties[i], score: sim})
		}
	}

	// Sort by similarity descending.
	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	filtered := make([]engine.BountyListing, len(results))
	for i, r := range results {
		filtered[i] = r.bounty
	}

	return engine.BountySearchOutput{
		Query:    query,
		Bounties: filtered,
	}, nil
}

// bountyLLMFallback uses the existing LLM summarization pipeline.
func bountyLLMFallback(ctx context.Context, input engine.BountySearchInput, bounties []engine.BountyListing, searxResults []engine.SearxngResult, contents map[string]string) (*mcp.CallToolResult, engine.SmartSearchOutput, error) {
	bountyOut, err := jobs.SummarizeBountyResults(ctx, input.Query, engine.BountySearchInstruction, 4000, searxResults, contents)
	if err != nil {
		slog.Warn("bounty_search: LLM summarization failed, returning raw", slog.Any("error", err))
		out := engine.BountySearchOutput{
			Query:    input.Query,
			Bounties: bounties,
			Summary:  fmt.Sprintf("Found %d bounties (LLM summary unavailable).", len(bounties)),
		}
		return bountyResult(out)
	}

	for i := range bountyOut.Bounties {
		b := &bountyOut.Bounties[i]
		if b.URL == "" && i < len(searxResults) {
			b.URL = searxResults[i].URL
		}
		if b.Source == "" {
			b.Source = "algora"
		}
	}
	return bountyResult(*bountyOut)
}

func bountyResult(out engine.BountySearchOutput) (*mcp.CallToolResult, engine.SmartSearchOutput, error) {
	jsonBytes, err := json.Marshal(out)
	if err != nil {
		return nil, engine.SmartSearchOutput{}, errors.New("json marshal failed")
	}
	result := engine.SmartSearchOutput{
		Query:   out.Query,
		Answer:  string(jsonBytes),
		Sources: []engine.SourceItem{},
	}
	return nil, result, nil
}
```

**Step 5: Build**

Run: `cd /home/krolik/src/go-job && go build -buildvcs=false ./...`
Expected: clean build

**Step 6: Commit**

```bash
cd /home/krolik/src/go-job
git add internal/jobserver/tool_bounty.go internal/engine/jobs/algora.go internal/engine/jobs/algora_test.go
git commit -m "feat: embedding-based bounty matching with LLM fallback"
```

---

### Task 4: Deploy & Test

**Files:**
- Modify: `/home/krolik/deploy/krolik-server/compose/apps.yml` (add EMBED_URL env)

**Step 1: Add EMBED_URL to go-job in compose**

In `/home/krolik/deploy/krolik-server/compose/apps.yml`, in go-job environment section add:

```yaml
- EMBED_URL=http://embed-server:8082
- MEMDB_URL=http://memdb-go:8080
```

**Step 2: Build and deploy**

```bash
cd /home/krolik/deploy/krolik-server/compose
docker compose build go-job
docker compose up -d go-job
```

**Step 3: Flush bounty cache and reconnect MCP**

```bash
docker exec redis redis-cli --scan --pattern "gj:*" | xargs -r -I{} docker exec redis redis-cli DEL "{}"
docker compose restart go-job
```

User: reconnect MCP via `/mcp`

**Step 4: Test empty query**

Call: `bounty_search(query="")` — should return all open bounties sorted by amount descending.

**Step 5: Test query filtering**

Call: `bounty_search(query="MCP")` — should return only MCP-related bounties via embedding similarity.

Call: `bounty_search(query="Scala")` — should return ZIO/Scala bounties.

**Step 6: Test fallback**

Stop embed-server temporarily:
```bash
docker compose stop embed-server
```

Call: `bounty_search(query="MCP")` — should fall back to LLM pipeline.

Restart:
```bash
docker compose start embed-server
```
