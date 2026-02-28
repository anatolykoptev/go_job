# Craigslist RSS Scraper Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace SearXNG-based Craigslist search with direct RSS feed scraping via BrowserClient for higher quality and quantity of job listings.

**Architecture:** Fetch Craigslist's built-in RSS feed (`?format=rss` on search URLs) using BrowserClient (Chrome TLS fingerprint) to bypass bot detection. Parse the RSS XML into `[]engine.SearxngResult`. Map user location strings to Craigslist region subdomains. Keep SearXNG as fallback when BrowserClient is unavailable.

**Tech Stack:** `encoding/xml` for RSS parsing, `engine.BrowserClient` for HTTP, existing `engine.SearxngResult` type.

---

### Task 1: Add Craigslist RSS types and region mapping

**Files:**
- Modify: `internal/engine/jobs/craigslist.go`

**Step 1: Add RSS XML types and region map**

Add after the existing imports and constants:

```go
// craigslistRSS represents the RSS feed from Craigslist search.
type craigslistRSS struct {
	XMLName xml.Name             `xml:"rss"`
	Channel craigslistRSSChannel `xml:"channel"`
}

type craigslistRSSChannel struct {
	Items []craigslistRSSItem `xml:"item"`
}

type craigslistRSSItem struct {
	Title       string `xml:"title"`
	Link        string `xml:"link"`
	Description string `xml:"description"`
	Date        string `xml:"dc:date"` // ISO 8601: 2026-02-28T12:00:00-08:00
}

// craigslistRegions maps common location keywords to Craigslist subdomains.
// Falls back to "www" (redirects to nearest region).
var craigslistRegions = map[string]string{
	"san francisco": "sfbay", "sf": "sfbay", "bay area": "sfbay", "oakland": "sfbay",
	"san jose": "sfbay", "silicon valley": "sfbay",
	"new york": "newyork", "nyc": "newyork", "manhattan": "newyork", "brooklyn": "newyork",
	"los angeles": "losangeles", "la": "losangeles",
	"chicago": "chicago",
	"seattle": "seattle", "tacoma": "seattle",
	"boston": "boston",
	"denver": "denver",
	"austin": "austin",
	"portland": "portland",
	"dallas": "dallas", "fort worth": "dallas",
	"houston": "houston",
	"atlanta": "atlanta",
	"miami": "miami",
	"phoenix": "phoenix",
	"philadelphia": "philadelphia", "philly": "philadelphia",
	"detroit": "detroit",
	"minneapolis": "minneapolis",
	"san diego": "sandiego",
	"washington": "washingtondc", "dc": "washingtondc",
	"las vegas": "lasvegas", "vegas": "lasvegas",
}

// resolveRegion maps a user-provided location string to a Craigslist subdomain.
func resolveRegion(location string) string {
	loc := strings.ToLower(strings.TrimSpace(location))
	if region, ok := craigslistRegions[loc]; ok {
		return region
	}
	// Try partial match: "San Francisco, CA" → "san francisco"
	for key, region := range craigslistRegions {
		if strings.Contains(loc, key) {
			return region
		}
	}
	return "www"
}
```

Add `"encoding/xml"` to the imports.

**Step 2: Build and verify**

Run: `cd ~/src/go-job && go build -buildvcs=false ./...`
Expected: clean build

**Step 3: Commit**

```bash
git add internal/engine/jobs/craigslist.go
git commit -m "feat(craigslist): add RSS types and region mapping"
```

---

### Task 2: Implement RSS feed fetcher

**Files:**
- Modify: `internal/engine/jobs/craigslist.go`

**Step 1: Add the RSS fetch + parse function**

Add a new function below `resolveRegion`:

```go
// fetchCraigslistRSS fetches and parses the Craigslist RSS feed for a given query/location.
// Requires BrowserClient (Craigslist blocks non-browser TLS fingerprints).
func fetchCraigslistRSS(ctx context.Context, query, location string, limit int) ([]engine.SearxngResult, error) {
	region := resolveRegion(location)
	feedURL := fmt.Sprintf("https://%s.craigslist.org/search/jjj?query=%s&format=rss",
		region, url.QueryEscape(query))

	ctx, cancel := context.WithTimeout(ctx, engine.Cfg.FetchTimeout)
	defer cancel()

	headers := engine.ChromeHeaders()
	headers["accept"] = "application/rss+xml, application/xml, text/xml"

	data, err := engine.RetryDo(ctx, engine.DefaultRetryConfig, func() ([]byte, error) {
		d, _, status, e := engine.Cfg.BrowserClient.Do("GET", feedURL, headers, nil)
		if e != nil {
			return nil, e
		}
		if status != http.StatusOK {
			return nil, fmt.Errorf("craigslist RSS status %d", status)
		}
		return d, nil
	})
	if err != nil {
		return nil, fmt.Errorf("craigslist RSS fetch: %w", err)
	}

	return parseCraigslistRSS(data, limit)
}

// parseCraigslistRSS parses Craigslist RSS XML into SearxngResult slice.
func parseCraigslistRSS(body []byte, limit int) ([]engine.SearxngResult, error) {
	var rss craigslistRSS
	if err := xml.Unmarshal(body, &rss); err != nil {
		return nil, fmt.Errorf("craigslist RSS parse: %w", err)
	}

	var results []engine.SearxngResult
	for _, item := range rss.Channel.Items {
		if item.Title == "" || item.Link == "" {
			continue
		}

		posted := ""
		if len(item.Date) >= 10 {
			posted = item.Date[:10] // YYYY-MM-DD
		}

		content := "**Source:** Craigslist"
		if posted != "" {
			content += " | **Posted:** " + posted
		}
		if item.Description != "" {
			desc := engine.TruncateAtWord(item.Description, 300)
			content += "\n\n" + desc
		}

		results = append(results, engine.SearxngResult{
			Title:   item.Title,
			Content: content,
			URL:     item.Link,
			Score:   0.8,
		})

		if len(results) >= limit {
			break
		}
	}

	return results, nil
}
```

Add `"fmt"`, `"net/http"`, `"net/url"` to imports.

**Step 2: Build and verify**

Run: `cd ~/src/go-job && go build -buildvcs=false ./...`
Expected: clean build

**Step 3: Commit**

```bash
git add internal/engine/jobs/craigslist.go
git commit -m "feat(craigslist): implement RSS feed fetcher via BrowserClient"
```

---

### Task 3: Wire RSS into SearchCraigslistJobs with SearXNG fallback

**Files:**
- Modify: `internal/engine/jobs/craigslist.go`

**Step 1: Rewrite SearchCraigslistJobs to try RSS first, fallback to SearXNG**

Replace the existing `SearchCraigslistJobs` function body:

```go
func SearchCraigslistJobs(ctx context.Context, query, location string, limit int) ([]engine.SearxngResult, error) {
	engine.IncrCraigslistRequests()

	// Primary: RSS feed via BrowserClient (structured data, more results).
	if engine.Cfg.BrowserClient != nil {
		results, err := fetchCraigslistRSS(ctx, query, location, limit)
		if err != nil {
			slog.Warn("craigslist: RSS fetch failed, falling back to SearXNG",
				slog.Any("error", err))
		} else if len(results) > 0 {
			slog.Debug("craigslist: RSS search complete", slog.Int("results", len(results)))
			return results, nil
		}
	}

	// Fallback: SearXNG site: search.
	searxQuery := query + " jobs " + craigslistSiteSearch
	if location != "" {
		searxQuery = query + " " + location + " jobs " + craigslistSiteSearch
	}

	searxResults, err := engine.SearchSearXNG(ctx, searxQuery, "en", "", engine.DefaultSearchEngine)
	if err != nil {
		slog.Warn("craigslist: SearXNG error", slog.Any("error", err))
	}

	var results []engine.SearxngResult
	for _, r := range searxResults {
		if !craigslistListingRe.MatchString(r.URL) {
			continue
		}
		if !isCraigslistJobCategory(r.URL) {
			continue
		}
		r.Content = "**Source:** Craigslist\n\n" + r.Content
		r.Score = 0.7
		results = append(results, r)
	}

	if len(results) > limit {
		results = results[:limit]
	}

	slog.Debug("craigslist: SearXNG fallback complete",
		slog.Int("raw", len(searxResults)),
		slog.Int("listings", len(results)))
	return results, nil
}
```

**Step 2: Build and vet**

Run: `cd ~/src/go-job && go build -buildvcs=false ./... && go vet ./...`
Expected: clean

**Step 3: Commit**

```bash
git add internal/engine/jobs/craigslist.go
git commit -m "feat(craigslist): use RSS as primary source, SearXNG as fallback"
```

---

### Task 4: Deploy and smoke test

**Files:** none (deployment only)

**Step 1: Fix file ownership**

```bash
sudo chown krolik:krolik ~/src/go-job/internal/engine/jobs/craigslist.go
```

**Step 2: Build and deploy**

```bash
cd ~/deploy/krolik-server
docker compose build --no-cache go-job
docker compose up -d --no-deps --force-recreate go-job
sleep 3
curl -s http://127.0.0.1:8891/health
```

Expected: `{"status":"ok",...}`

**Step 3: Smoke test — RSS path**

```bash
curl -s -X POST http://127.0.0.1:8891/mcp -H 'Content-Type: application/json' -d '{
  "jsonrpc":"2.0","id":1,"method":"tools/call",
  "params":{"name":"job_search","arguments":{"query":"software engineer","location":"San Francisco","platform":"craigslist","limit":5}}
}'
```

Expected: jobs with `craigslist.org` URLs, more results than before.

**Step 4: Smoke test — different regions**

Test with: `"driver" + "Los Angeles"`, `"nurse" + "Seattle"`, `"sales" + "New York"`

**Step 5: Verify metrics**

```bash
curl -s http://127.0.0.1:8891/metrics | grep craigslist
```

Expected: `craigslist_requests` incremented.

**Step 6: Commit and push**

```bash
cd ~/src/go-job
git add -A
git commit -m "deploy: verify craigslist RSS scraper"
git push origin master
```
