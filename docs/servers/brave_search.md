# Brave Search MCP Server

## Overview

The Brave Search MCP server integrates with the Brave Search API to provide comprehensive search capabilities across multiple content types. It enables programmatic access to Brave's privacy-focused search engine with support for web, local, news, images, videos, and AI-powered summarization.

**Transport:** stdio (local process, default) or HTTP

## What it Does

This server enables:
- Performing web searches with advanced filtering and rich result types
- Searching for local businesses and services with ratings and AI descriptions
- Searching news articles with time-based filtering
- Finding images and videos with metadata
- Generating AI-powered summaries of search results
- Controlling result freshness, safety levels, and language preferences
- Paginating through search results

## Tools

The Brave Search server provides the following tools:

### `brave_web_search`
Performs comprehensive web searches with rich result types and advanced filtering.

**Parameters:**
- `query` (string, required): Search terms (max 400 characters or 50 words)
- `country` (string, optional): Country code (default: "US")
- `search_lang` (string, optional): Search language (default: "en")
- `count` (number, optional): Results per page, 1-20 (default: 10)
- `offset` (number, optional): Pagination offset, max 9 (default: 0)
- `safesearch` (string, optional): Content filtering - "off", "moderate", or "strict"
- `freshness` (string, optional): Time-based filtering - "pd" (day), "pw" (week), "pm" (month), "py" (year), or date range
- `summary` (boolean, optional): Enable to get AI summarization key for use with `brave_summarizer`

**Returns:** Search results including titles, URLs, descriptions, rich metadata, and optional summary key

### `brave_local_search`
Searches for local businesses and places with ratings, hours, and AI descriptions.

**Parameters:**
- `query` (string, required): Local search terms (e.g., "coffee shops near me")
- Same filtering parameters as `brave_web_search` (country, count, safesearch, etc.)

**Returns:** Local business information including names, addresses, phone numbers, ratings, hours, and AI-generated descriptions

**Note:** Requires Pro plan. Automatically falls back to web search if no local results are found.

### `brave_video_search`
Searches for videos with metadata and thumbnails.

**Parameters:**
- `query` (string, required): Video search terms
- `count` (number, optional): Results per page, 1-50 (default: 20)
- `safesearch` (string, optional): Content filtering
- `freshness` (string, optional): Time-based filtering
- `spellcheck` (boolean, optional): Enable spell checking

**Returns:** Video results with titles, URLs, thumbnails, duration, and metadata

### `brave_image_search`
Searches for images with filtering options.

**Parameters:**
- `query` (string, required): Image search terms
- `count` (number, optional): Results per page, 1-200 (default: 50)
- `safesearch` (string, optional): Content filtering (default: "strict")

**Returns:** Image results with URLs, thumbnails, dimensions, and source information

### `brave_news_search`
Searches current news articles with freshness controls.

**Parameters:**
- `query` (string, required): News search terms
- `count` (number, optional): Results per page, 1-50 (default: 20)
- `freshness` (string, optional): Time-based filtering (default: "pd" for last 24 hours)

**Returns:** News articles with titles, URLs, descriptions, publication dates, and source information

### `brave_summarizer`
Generates AI-powered summaries from web search results.

**Parameters:**
- `key` (string, required): Summary key obtained from `brave_web_search` with `summary: true`
- `entity_info` (boolean, optional): Include entity information in the summary
- `inline_references` (boolean, optional): Add source URL references within the summary

**Returns:** AI-generated summary with optional entity information and inline citations

**Note:** Requires Pro plan. Must first perform a web search with `summary: true` to obtain the summary key.

## Configuration

### Required Tokens

The Brave Search server requires a Brave Search API key.

**How to obtain:**
1. Sign up at [brave.com/search/api](https://brave.com/search/api/)
2. Access the developer dashboard
3. Generate an API key
4. Copy the API key for use in configuration

**Plan tiers:**
- **Free plan:** 2,000 queries/month, basic web search
- **Pro plan:** Enhanced features including local search, AI summaries, extra snippets, and additional data fields

### Token Configuration

**Default Token Name:** `BRAVE_API_KEY`

**Authentication Type:** API Key (Bearer Token)

**OAuth Support:** No - Brave Search uses API keys for authentication.

**Token Storage:**

The recommended approach is to store the token in the MCP CLI token store and use token replacement:

```bash
# Store the token interactively (recommended - no shell history)
uv run mcp-cli token set brave_search --type bearer
# Enter token value for 'brave_search': [hidden input]

# Or provide directly (less secure - visible in shell history)
uv run mcp-cli token set brave_search --type bearer --value "your_api_key_here"
```

**Alternative: Environment Variable**

You can also set it as an environment variable:
```bash
export BRAVE_API_KEY="your_api_key_here"
```

### Example Configuration

**With Token Store (Recommended - Secure Token Replacement):**

MCP CLI supports automatic token replacement using the `${TOKEN:namespace:name}` syntax. This keeps your API keys secure and out of config files:

```json
{
  "mcpServers": {
    "brave_search": {
      "command": "npx",
      "args": ["-y", "@brave/brave-search-mcp-server"],
      "env": {
        "BRAVE_API_KEY": "${TOKEN:bearer:brave_search}"
      }
    }
  }
}
```

The `${TOKEN:bearer:brave_search}` placeholder will be automatically replaced with the actual token value from your secure token store at runtime.

**With Environment Variable:**
```json
{
  "mcpServers": {
    "brave_search": {
      "command": "npx",
      "args": ["-y", "@brave/brave-search-mcp-server"],
      "env": {
        "BRAVE_API_KEY": "${BRAVE_API_KEY}"
      }
    }
  }
}
```

## Usage Notes

- This server runs locally via `npx` (no persistent installation required)
- Default transport is stdio in version 2.x; HTTP transport also available
- The first run will download the package automatically
- API rate limits apply based on your plan:
  - Free: 2,000 queries/month, basic web search only
  - Pro: Enhanced features with local search, news, images, videos, and AI summaries
- Search queries are processed through Brave's privacy-focused search engine
- Results respect Brave's content safety and filtering policies
- Query length limited to 400 characters or 50 words
- Pagination offset limited to 9 for web searches
- Image search can return up to 200 results per query
- Video and news search can return up to 50 results per query
- AI summarization requires Pro plan and a summary key from web search

### Advanced Configuration

The server supports additional environment variables for customization:
- `BRAVE_MCP_TRANSPORT`: Set to "http" or "stdio" (default: "stdio")
- `BRAVE_MCP_PORT`: HTTP server port (default: 8080, only for HTTP transport)
- `BRAVE_MCP_HOST`: HTTP server host (default: "0.0.0.0", only for HTTP transport)
- `BRAVE_MCP_LOG_LEVEL`: Logging verbosity control
- `BRAVE_MCP_ENABLED_TOOLS`: Whitelist specific tools (comma-separated)
- `BRAVE_MCP_DISABLED_TOOLS`: Blacklist specific tools (comma-separated)

## Token Management

See the [TOKEN_MANAGEMENT.md](../TOKEN_MANAGEMENT.md) documentation for details on:
- Storing tokens securely using the token store
- Using token replacement syntax in configuration
- Managing multiple API keys
- Rotating tokens when needed
