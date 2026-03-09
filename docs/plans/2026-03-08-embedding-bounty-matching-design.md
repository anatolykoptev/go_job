# Embedding-Based Bounty Matching Design

## Goal
Replace LLM summarization in bounty_search with direct embedding similarity via embed-server, keeping LLM as fallback.

## Architecture
Pipeline: scrape algora → filter open (GitHub API) → fetch issue content → embed bounties + query → cosine similarity → sort → return. If embed-server unavailable, fallback to current LLM pipeline.

## Components

### 1. Embed Client (`internal/engine/jobs/embed.go`)
- HTTP client to `embed-server:8082/v1/embeddings` (OpenAI-compatible)
- Batch embedding (all bounties in one request)
- Returns `[][]float32` (1024-dim vectors)
- Config: `EMBED_URL` env var (default: `http://embed-server:8082`)

### 2. Skills Extraction (in `algora.go`)
- GitHub API: issue labels + repo primary language
- Keyword matching: dictionary of ~50 technologies, regex on title + issue body
- Deduplicate, sort alphabetically

### 3. Cosine Similarity (in `embed.go`)
- `cosineSimilarity(a, b []float32) float32`
- Threshold: 0.3 for e5-large normalized vectors

### 4. Updated Pipeline (`tool_bounty.go`)
- If `query != ""`: embed query + embed bounties → sort by similarity → filter below threshold
- If `query == ""`: return all, sorted by amount (descending)
- Summary field removed — client interprets results directly
- On embed-server failure: fallback to LLM pipeline (existing code)

### What stays
- `SummarizeBountyResults()`, `BountiesToSearxngResults()`, `llmBountyOutput`, `BountySearchInstruction` — all remain as LLM fallback path
