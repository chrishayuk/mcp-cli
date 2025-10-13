#!/usr/bin/env python3
"""
MCP-CLI Complete Streaming Demo
================================
Shows all content types: code, tools, tables, markdown, and mixed content.
"""

import asyncio
import random
from rich.console import Console
from chuk_term.ui.output import get_output
from chuk_term.ui.theme import set_theme

# Import streaming helpers from MCP-CLI
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_cli.ui.streaming_display import StreamingContext, tokenize_text


async def stream_response(text: str, speed: float = 0.02):
    """Simulate streaming with variable speed."""
    for token in tokenize_text(text):
        yield token
        await asyncio.sleep(speed * random.uniform(0.8, 1.2))


async def main():
    """Run complete streaming demo."""
    output = get_output()
    set_theme("default")
    console = Console()

    # Header
    output.print("\n" + "=" * 70)
    output.print("MCP-CLI COMPLETE STREAMING DEMO", style="bold cyan")
    output.print("Showcasing all content types", style="dim")
    output.print("=" * 70)

    # 1. Code Generation
    output.print("\n" + "─" * 40)
    output.print("1. CODE GENERATION", style="bold yellow")
    output.print("─" * 40)

    code_response = """I'll help you create a Python web scraper with error handling.

```python
import requests
from bs4 import BeautifulSoup
import time
from typing import List, Dict, Optional
import logging

class WebScraper:
    def __init__(self, base_url: str, delay: float = 1.0):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
    def fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None
        finally:
            time.sleep(self.delay)  # Rate limiting
    
    def extract_links(self, soup: BeautifulSoup) -> List[str]:
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http'):
                links.append(href)
            elif href.startswith('/'):
                links.append(self.base_url + href)
        return links
    
    def scrape_content(self, url: str) -> Dict[str, any]:
        soup = self.fetch_page(url)
        if not soup:
            return {}
        
        return {
            'title': soup.find('title').text if soup.find('title') else '',
            'links': self.extract_links(soup),
            'text': soup.get_text(strip=True)[:500]  # First 500 chars
        }

# Usage
scraper = WebScraper('https://example.com')
data = scraper.scrape_content('https://example.com/page')
print(f"Found {len(data.get('links', []))} links")
```

This scraper includes rate limiting, error handling, and session management."""

    with StreamingContext(
        console=console,
        title="🐍 Python Web Scraper",
        mode="response",
        refresh_per_second=10,
        transient=True,
    ) as ctx:
        async for chunk in stream_response(code_response, 0.015):
            ctx.update(chunk)

    await asyncio.sleep(1)

    # 2. Tool Execution
    output.print("\n" + "─" * 40)
    output.print("2. TOOL EXECUTION", style="bold yellow")
    output.print("─" * 40)

    tool_response = """Analyzing git repository statistics...

Scanning commit history...
Found 1,247 commits across 23 branches

Top contributors:
1. Alice Johnson - 423 commits (33.9%)
2. Bob Smith - 312 commits (25.0%)
3. Carol Davis - 198 commits (15.9%)
4. David Wilson - 187 commits (15.0%)
5. Others - 127 commits (10.2%)

File statistics:
- Total files: 892
- Lines of code: 45,678
- Test coverage: 87.3%

Recent activity (last 30 days):
- Commits: 89
- Pull requests: 23
- Issues closed: 41
- Active contributors: 8

Repository health score: 92/100 ✅"""

    with StreamingContext(
        console=console,
        title="📊 Git Analyzer",
        mode="tool",
        refresh_per_second=8,
        transient=True,
    ) as ctx:
        async for chunk in stream_response(tool_response, 0.02):
            ctx.update(chunk)

    await asyncio.sleep(1)

    # 3. Markdown Tables
    output.print("\n" + "─" * 40)
    output.print("3. MARKDOWN TABLES", style="bold yellow")
    output.print("─" * 40)

    table_response = """Here's a performance comparison of different algorithms:

## Algorithm Performance Analysis

| Algorithm | Time Complexity | Space | 1K items | 100K items | 1M items | Best For |
|-----------|----------------|-------|----------|------------|----------|----------|
| QuickSort | O(n log n) | O(log n) | 0.3ms | 42ms | 520ms | General use |
| MergeSort | O(n log n) | O(n) | 0.4ms | 48ms | 580ms | Stable sort |
| HeapSort | O(n log n) | O(1) | 0.5ms | 61ms | 710ms | Memory constrained |
| TimSort | O(n log n) | O(n) | 0.2ms | 38ms | 450ms | Real-world data |
| RadixSort | O(nk) | O(n+k) | 0.1ms | 15ms | 180ms | Integer sorting |

### Database Query Performance

| Database | Simple Query | Complex Join | Aggregation | Full-text Search |
|----------|-------------|--------------|-------------|------------------|
| PostgreSQL | 1.2ms | 15ms | 45ms | 89ms |
| MySQL | 1.5ms | 18ms | 52ms | 123ms |
| MongoDB | 2.1ms | N/A | 38ms | 156ms |
| Redis | 0.3ms | N/A | 8ms | N/A |
| Elasticsearch | 3.5ms | N/A | 22ms | 12ms ⭐ |

**Key Findings:**
- TimSort excels with partially sorted data
- Redis is unmatched for simple key-value operations
- Elasticsearch dominates full-text search
- PostgreSQL offers best all-around performance"""

    with StreamingContext(
        console=console,
        title="📈 Performance Analysis",
        mode="response",
        refresh_per_second=8,
        transient=True,
    ) as ctx:
        async for chunk in stream_response(table_response, 0.018):
            ctx.update(chunk)

    await asyncio.sleep(1)

    # 4. Thinking Process
    output.print("\n" + "─" * 40)
    output.print("4. THINKING PROCESS", style="bold yellow")
    output.print("─" * 40)

    thinking_response = """Let me analyze this system design problem step by step.

The requirements are:
- Handle 1 million concurrent users
- Sub-100ms response time
- 99.99% uptime
- Global distribution

Considering the scale, I need to think about:

1. Load balancing strategy
   - Geographic distribution is key
   - Need multiple regions with failover
   - Consider anycast routing for lowest latency

2. Database architecture
   - Single database won't scale
   - Need sharding or federation
   - Read replicas in each region
   - Consider eventual consistency tradeoffs

3. Caching layers
   - CDN for static content (obviously)
   - Application-level caching with Redis
   - Database query result caching
   - Edge caching for API responses

4. Service architecture
   - Microservices for independent scaling
   - Service mesh for observability
   - Circuit breakers for resilience
   - Message queues for async processing

Given these constraints, I recommend:
- Multi-region deployment with active-active setup
- Cassandra for distributed data with local quorum reads
- Redis for session management and caching
- Kubernetes for orchestration
- Istio for service mesh capabilities

This architecture should handle the load while maintaining low latency."""

    with StreamingContext(
        console=console,
        title="💭 System Design Analysis",
        mode="thinking",
        refresh_per_second=8,
        transient=True,
    ) as ctx:
        async for chunk in stream_response(thinking_response, 0.022):
            ctx.update(chunk)

    await asyncio.sleep(1)

    # 5. Mixed Content
    output.print("\n" + "─" * 40)
    output.print("5. MIXED CONTENT", style="bold yellow")
    output.print("─" * 40)

    mixed_response = """I'll analyze your API and provide optimization recommendations.

## Current API Performance

Your REST API is handling 5,000 requests/second with these metrics:

| Endpoint | Avg Response | P99 Latency | Error Rate | Cache Hit |
|----------|-------------|-------------|------------|-----------|
| /users | 45ms | 234ms | 0.1% | 78% |
| /products | 67ms | 345ms | 0.2% | 65% |
| /search | 123ms | 567ms | 0.5% | 42% |
| /checkout | 234ms | 890ms | 1.2% | N/A |

### Issues Identified

1. **Checkout endpoint is too slow** - Here's the problematic code:

```javascript
// Current implementation (inefficient)
async function checkout(cartId) {
  const cart = await db.query('SELECT * FROM carts WHERE id = ?', [cartId]);
  
  for (const item of cart.items) {
    const product = await db.query('SELECT * FROM products WHERE id = ?', [item.id]);
    const inventory = await db.query('SELECT * FROM inventory WHERE product_id = ?', [item.id]);
    // N+1 query problem!
  }
  
  return processPayment(cart);
}

// Optimized implementation
async function checkout(cartId) {
  const cart = await db.query(`
    SELECT c.*, p.*, i.quantity as stock
    FROM carts c
    JOIN cart_items ci ON c.id = ci.cart_id
    JOIN products p ON ci.product_id = p.id
    JOIN inventory i ON p.id = i.product_id
    WHERE c.id = ?
  `, [cartId]);
  
  return processPayment(cart);
}
```

2. **Search needs Elasticsearch** - Current LIKE queries won't scale

3. **Add response caching** - Use Redis with appropriate TTLs:

```python
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_response(ttl=300):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{str(args)}:{str(kwargs)}"
            
            # Try cache first
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Cache miss - execute function
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@cache_response(ttl=600)
async def get_products(category_id):
    return await db.fetch_products(category_id)
```

### Optimization Roadmap

| Priority | Task | Impact | Effort | Timeline |
|----------|------|--------|--------|----------|
| 🔴 High | Fix N+1 queries | -60% latency | Low | 1 week |
| 🔴 High | Add Redis caching | -40% load | Medium | 2 weeks |
| 🟠 Medium | Implement Elasticsearch | -70% search time | High | 1 month |
| 🟡 Low | Add CDN | -30% bandwidth | Low | 1 week |

Implementing these changes should reduce P99 latency by 65% and increase throughput to 12,000 req/s."""

    with StreamingContext(
        console=console,
        title="🔧 API Optimization Analysis",
        mode="response",
        refresh_per_second=10,
        transient=True,
    ) as ctx:
        async for chunk in stream_response(mixed_response, 0.012):
            ctx.update(chunk)

    # Summary
    output.print("\n" + "=" * 70)
    output.success("✅ Complete Streaming Demo Finished!")
    output.print("=" * 70)

    output.info("\n📊 Demonstrated:")
    output.print("• Code generation with syntax highlighting")
    output.print("• Tool execution with progress tracking")
    output.print("• Markdown tables with data")
    output.print("• Thinking/reasoning process")
    output.print("• Mixed content with code, tables, and analysis")

    output.print("\n✨ All content types handled seamlessly with:")
    output.print("• Automatic content detection")
    output.print("• Appropriate phase messages")
    output.print("• Clean progressive display")
    output.print("• Beautiful final formatting")


if __name__ == "__main__":
    output = get_output()
    output.print("\n🚀 Starting Complete Streaming Demo...")
    output.print("Showcasing code, tools, tables, and mixed content\n")
    asyncio.run(main())
