#!/usr/bin/env python3
"""
MCP-CLI Streaming Showcase - Full Demo
=======================================
Comprehensive demonstration of all streaming capabilities including:
- Code generation with syntax highlighting
- Tool execution with progress
- Markdown tables
- Regular content
- Mixed content types
- Error handling
- Real-world LLM response patterns
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


async def stream_response(text: str, speed: float = 0.025):
    """Simulate streaming an LLM response with configurable speed."""
    for token in tokenize_text(text):
        yield token
        # Add some variability to simulate network latency
        delay = speed * random.uniform(0.8, 1.2)
        await asyncio.sleep(delay)


async def demo_code_generation():
    """Demonstrate code generation with multiple languages."""
    output = get_output()
    console = Console()
    
    output.print("\n" + "─" * 40)
    output.print("Code Generation Examples", style="bold cyan")
    output.print("─" * 40)
    
    # Python code example
    output.info("Generating Python implementation...")
    
    python_code = """I'll create a comprehensive Python class for data processing.

## DataProcessor Class

Here's a full-featured implementation with error handling:

```python
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import logging

@dataclass
class ProcessingConfig:
    \"\"\"Configuration for data processing.\"\"\"
    chunk_size: int = 1000
    validate: bool = True
    normalize: bool = False
    fill_missing: Optional[Any] = None

class DataProcessor:
    \"\"\"Advanced data processing pipeline.\"\"\"
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.logger = logging.getLogger(__name__)
        self._cache = {}
        
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Main processing pipeline.\"\"\"
        self.logger.info(f"Processing {len(data)} rows")
        
        # Validation
        if self.config.validate:
            self._validate_data(data)
        
        # Chunked processing for large datasets
        if len(data) > self.config.chunk_size:
            return self._process_chunked(data)
        
        # Apply transformations
        data = self._clean_data(data)
        
        if self.config.normalize:
            data = self._normalize_data(data)
            
        if self.config.fill_missing is not None:
            data = data.fillna(self.config.fill_missing)
        
        return data
    
    def _validate_data(self, data: pd.DataFrame) -> None:
        \"\"\"Validate input data.\"\"\"
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Check for required columns
        required_cols = ['id', 'timestamp', 'value']
        missing = set(required_cols) - set(data.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
    
    def _process_chunked(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Process large datasets in chunks.\"\"\"
        chunks = []
        for i in range(0, len(data), self.config.chunk_size):
            chunk = data.iloc[i:i + self.config.chunk_size]
            processed = self.process(chunk)
            chunks.append(processed)
        return pd.concat(chunks, ignore_index=True)
    
    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Clean and prepare data.\"\"\"
        # Remove duplicates
        data = data.drop_duplicates()
        
        # Convert data types
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        return data
    
    def _normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"Normalize numerical columns.\"\"\"
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = (data[numeric_cols] - data[numeric_cols].mean()) / data[numeric_cols].std()
        return data

# Usage example
processor = DataProcessor(ProcessingConfig(
    chunk_size=500,
    validate=True,
    normalize=True,
    fill_missing=0
))

df = pd.DataFrame({
    'id': range(1000),
    'timestamp': pd.date_range('2024-01-01', periods=1000, freq='H'),
    'value': np.random.randn(1000)
})

result = processor.process(df)
print(f"Processed {len(result)} rows successfully")
```

This implementation includes:
- Configuration management with dataclasses
- Comprehensive error handling
- Chunked processing for scalability
- Data validation and cleaning
- Normalization capabilities
- Logging integration"""
    
    with StreamingContext(
        console=console,
        title="🐍 Python Code",
        mode="response",
        refresh_per_second=10,
        transient=True
    ) as ctx:
        async for chunk in stream_response(python_code, speed=0.02):
            ctx.update(chunk)
    
    await asyncio.sleep(1)
    
    # JavaScript/TypeScript example
    output.info("\nGenerating TypeScript implementation...")
    
    typescript_code = """Here's the equivalent TypeScript implementation:

```typescript
interface ProcessingConfig {
  chunkSize?: number;
  validate?: boolean;
  normalize?: boolean;
  fillMissing?: any;
}

interface DataRow {
  id: number;
  timestamp: Date;
  value: number;
  [key: string]: any;
}

class DataProcessor {
  private config: Required<ProcessingConfig>;
  private cache: Map<string, any> = new Map();
  
  constructor(config?: ProcessingConfig) {
    this.config = {
      chunkSize: config?.chunkSize ?? 1000,
      validate: config?.validate ?? true,
      normalize: config?.normalize ?? false,
      fillMissing: config?.fillMissing ?? null
    };
  }
  
  async process(data: DataRow[]): Promise<DataRow[]> {
    console.log(`Processing ${data.length} rows`);
    
    if (this.config.validate) {
      this.validateData(data);
    }
    
    // Process in chunks for large datasets
    if (data.length > this.config.chunkSize) {
      return this.processChunked(data);
    }
    
    let processed = this.cleanData(data);
    
    if (this.config.normalize) {
      processed = this.normalizeData(processed);
    }
    
    return processed;
  }
  
  private validateData(data: DataRow[]): void {
    if (!data || data.length === 0) {
      throw new Error('Input data is empty');
    }
    
    const requiredFields = ['id', 'timestamp', 'value'];
    const firstRow = data[0];
    
    for (const field of requiredFields) {
      if (!(field in firstRow)) {
        throw new Error(`Missing required field: ${field}`);
      }
    }
  }
  
  private async processChunked(data: DataRow[]): Promise<DataRow[]> {
    const chunks: DataRow[][] = [];
    
    for (let i = 0; i < data.length; i += this.config.chunkSize) {
      const chunk = data.slice(i, i + this.config.chunkSize);
      const processed = await this.process(chunk);
      chunks.push(processed);
    }
    
    return chunks.flat();
  }
  
  private cleanData(data: DataRow[]): DataRow[] {
    // Remove duplicates based on id
    const seen = new Set<number>();
    return data.filter(row => {
      if (seen.has(row.id)) return false;
      seen.add(row.id);
      return true;
    });
  }
  
  private normalizeData(data: DataRow[]): DataRow[] {
    // Calculate statistics
    const values = data.map(row => row.value);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(
      values.reduce((sq, n) => sq + Math.pow(n - mean, 2), 0) / values.length
    );
    
    // Normalize
    return data.map(row => ({
      ...row,
      value: (row.value - mean) / std
    }));
  }
}

// Usage
const processor = new DataProcessor({
  chunkSize: 500,
  validate: true,
  normalize: true
});

const testData: DataRow[] = Array.from({ length: 1000 }, (_, i) => ({
  id: i,
  timestamp: new Date(Date.now() + i * 3600000),
  value: Math.random() * 100
}));

processor.process(testData).then(result => {
  console.log(`Processed ${result.length} rows successfully`);
});
```

The TypeScript version provides the same functionality with strong typing."""
    
    with StreamingContext(
        console=console,
        title="📘 TypeScript Code",
        mode="response",
        refresh_per_second=10,
        transient=True
    ) as ctx:
        async for chunk in stream_response(typescript_code, speed=0.015):
            ctx.update(chunk)


async def demo_tool_execution():
    """Demonstrate various tool executions."""
    output = get_output()
    console = Console()
    
    output.print("\n" + "─" * 40)
    output.print("Tool Execution Examples", style="bold cyan")
    output.print("─" * 40)
    
    # Database query tool
    output.info("Executing database analysis...")
    
    db_tool = """Analyzing database performance metrics...

Connecting to PostgreSQL database...
Connection established successfully.

Executing query to analyze table sizes:

```sql
SELECT 
    schemaname AS schema,
    tablename AS table,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
    n_live_tup AS row_count,
    n_dead_tup AS dead_rows,
    last_vacuum,
    last_autovacuum
FROM pg_stat_user_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
LIMIT 10;
```

Query execution time: 0.234s

Results retrieved:
- users: 1.2 GB, 2.5M rows
- orders: 890 MB, 1.8M rows  
- products: 456 MB, 500K rows
- sessions: 234 MB, 3.2M rows
- logs: 198 MB, 5.1M rows

Analyzing index usage...

```sql
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan as index_scans,
    idx_tup_read as tuples_read,
    idx_tup_fetch as tuples_fetched,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

Found 3 unused indexes consuming 45 MB of storage.

Recommendations:
1. Consider dropping unused indexes to save storage
2. The 'orders' table needs vacuum (15% dead tuples)
3. Add index on users.email for faster lookups

Total execution time: 1.47s"""
    
    with StreamingContext(
        console=console,
        title="🗄️ Database Analyzer",
        mode="tool",
        refresh_per_second=8,
        transient=True
    ) as ctx:
        async for chunk in stream_response(db_tool, speed=0.02):
            ctx.update(chunk)
    
    await asyncio.sleep(1)
    
    # File system tool
    output.info("\nScanning file system...")
    
    fs_tool = """Scanning project directory for large files...

Walking directory tree: /Users/project/src
Found 1,247 files in 89 directories

Analyzing file sizes and types:

Largest files:
1. node_modules/webpack/lib/webpack.js - 2.3 MB
2. dist/bundle.js - 1.8 MB
3. data/sample_dataset.json - 1.2 MB
4. logs/application.log - 987 KB
5. docs/api_reference.pdf - 876 KB

File type distribution:
- JavaScript: 423 files (34%)
- TypeScript: 312 files (25%)
- JSON: 156 files (13%)
- Python: 98 files (8%)
- Markdown: 87 files (7%)
- Other: 171 files (13%)

Checking for common issues:
✓ No .env files in repository
✓ No API keys found in code
⚠ Large log files detected (consider rotation)
⚠ node_modules included (add to .gitignore)

Scan completed in 0.89s"""
    
    with StreamingContext(
        console=console,
        title="📁 File System Scanner",
        mode="tool",
        refresh_per_second=8,
        transient=True
    ) as ctx:
        async for chunk in stream_response(fs_tool, speed=0.018):
            ctx.update(chunk)


async def demo_markdown_tables():
    """Demonstrate various markdown table formats."""
    output = get_output()
    console = Console()
    
    output.print("\n" + "─" * 40)
    output.print("Markdown Tables & Data Visualization", style="bold cyan")
    output.print("─" * 40)
    
    # Performance comparison table
    output.info("Generating performance comparison...")
    
    perf_table = """I'll analyze the performance metrics across different implementations.

## Performance Comparison Analysis

After running comprehensive benchmarks, here are the results:

### Execution Time Comparison

| Algorithm | Small Dataset (1K) | Medium Dataset (100K) | Large Dataset (10M) | Memory Usage |
|-----------|-------------------|-----------------------|---------------------|--------------|
| **QuickSort** | 0.003ms | 0.42ms | 89ms | O(log n) |
| **MergeSort** | 0.004ms | 0.51ms | 102ms | O(n) |
| **HeapSort** | 0.005ms | 0.63ms | 134ms | O(1) |
| **BubbleSort** | 0.089ms | 892ms | 🚫 Timeout | O(1) |
| **TimSort** ⭐ | 0.003ms | 0.38ms | 71ms | O(n) |

### Database Query Performance

| Query Type | PostgreSQL | MySQL | MongoDB | Redis | ElasticSearch |
|------------|------------|--------|---------|--------|---------------|
| Simple SELECT | 0.8ms | 1.2ms | 2.1ms | 0.3ms | 3.2ms |
| JOIN (2 tables) | 4.5ms | 6.2ms | N/A | N/A | N/A |
| Aggregation | 12ms | 15ms | 8ms | 1.2ms | 5ms |
| Full-text Search | 45ms | 67ms | 89ms | N/A | 12ms ⭐ |
| Geospatial | 23ms | 31ms | 18ms ⭐ | N/A | 28ms |
| Time-series | 8ms | 11ms | 14ms | 2ms ⭐ | 9ms |

### API Response Times by Region

| Endpoint | US-East | US-West | EU-West | AP-Southeast | AU-Sydney |
|----------|---------|---------|---------|--------------|-----------|
| `/api/users` | 45ms | 52ms | 78ms | 125ms | 142ms |
| `/api/products` | 38ms | 44ms | 69ms | 112ms | 128ms |
| `/api/search` | 89ms | 95ms | 134ms | 189ms | 203ms |
| `/api/checkout` | 234ms | 241ms | 287ms | 342ms | 358ms |
| **Average** | **101ms** | **108ms** | **142ms** | **192ms** | **208ms** |

### Framework Comparison

| Framework | Requests/sec | Latency (p99) | Memory | CPU Usage | Ease of Use |
|-----------|--------------|---------------|---------|-----------|-------------|
| **FastAPI** 🐍 | 15,234 | 89ms | 124MB | 45% | ⭐⭐⭐⭐⭐ |
| **Express.js** 🟢 | 12,456 | 102ms | 98MB | 38% | ⭐⭐⭐⭐ |
| **Gin** 🔷 | 28,901 | 34ms | 67MB | 32% | ⭐⭐⭐ |
| **Spring Boot** ☕ | 8,234 | 156ms | 512MB | 67% | ⭐⭐⭐ |
| **Django** 🐍 | 4,567 | 234ms | 234MB | 58% | ⭐⭐⭐⭐ |

**Key Findings:**
- TimSort performs best overall for mixed data patterns
- Redis excels at caching and time-series data
- MongoDB leads in geospatial queries
- Gin offers best raw performance but with steeper learning curve
- FastAPI provides excellent balance of performance and developer experience"""
    
    with StreamingContext(
        console=console,
        title="📊 Performance Analysis",
        mode="response",
        refresh_per_second=8,
        transient=True
    ) as ctx:
        async for chunk in stream_response(perf_table, speed=0.015):
            ctx.update(chunk)
    
    await asyncio.sleep(1)
    
    # Cost analysis table
    output.info("\nGenerating cost analysis...")
    
    cost_table = """## Cloud Services Cost Analysis

Based on current usage patterns, here's the monthly cost breakdown:

### AWS vs Azure vs GCP Comparison

| Service Category | AWS | Azure | GCP | Best Value |
|------------------|-----|-------|-----|------------|
| **Compute (4x m5.xlarge)** | $547.20 | $524.16 | $508.32 | GCP 💰 |
| **Storage (10TB S3/Blob)** | $230.40 | $208.90 | $200.00 | GCP 💰 |
| **Database (RDS/SQL)** | $892.80 | $856.32 | $823.68 | GCP 💰 |
| **CDN (CloudFront/CDN)** | $125.00 | $118.00 | $120.00 | Azure 💰 |
| **Load Balancer** | $25.00 | $23.00 | $18.00 | GCP 💰 |
| **Data Transfer (5TB)** | $450.00 | $425.00 | $390.00 | GCP 💰 |
| **Monitoring** | $89.00 | $79.00 | $95.00 | Azure 💰 |
| **Total Monthly** | **$2,359.40** | **$2,234.38** | **$2,155.00** | **GCP** ✅ |

Potential savings by switching to GCP: $204.40/month ($2,452.80/year)"""
    
    with StreamingContext(
        console=console,
        title="💰 Cost Analysis",
        mode="response",
        refresh_per_second=8,
        transient=True
    ) as ctx:
        async for chunk in stream_response(cost_table, speed=0.02):
            ctx.update(chunk)


async def demo_mixed_content():
    """Demonstrate mixed content types in a single response."""
    output = get_output()
    console = Console()
    
    output.print("\n" + "─" * 40)
    output.print("Mixed Content Response", style="bold cyan")
    output.print("─" * 40)
    output.info("Analyzing system architecture and providing recommendations...")
    
    mixed_response = """I'll analyze your system architecture and provide comprehensive recommendations.

## System Architecture Analysis

After reviewing your codebase and infrastructure, here's my analysis:

### Current Architecture Overview

Your system follows a microservices architecture with the following components:

```mermaid
graph TB
    Client[Web Client] --> LB[Load Balancer]
    LB --> API1[API Server 1]
    LB --> API2[API Server 2]
    LB --> API3[API Server 3]
    
    API1 --> Cache[Redis Cache]
    API2 --> Cache
    API3 --> Cache
    
    API1 --> DB[(PostgreSQL)]
    API2 --> DB
    API3 --> DB
    
    API1 --> Queue[RabbitMQ]
    API2 --> Queue
    API3 --> Queue
    
    Queue --> Worker1[Worker 1]
    Queue --> Worker2[Worker 2]
    
    Worker1 --> S3[S3 Storage]
    Worker2 --> S3
```

### Performance Metrics

Based on the last 30 days of monitoring data:

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| API Response Time (p95) | 234ms | <200ms | ⚠️ Needs Improvement |
| Database Query Time | 45ms | <50ms | ✅ Good |
| Cache Hit Rate | 67% | >80% | ⚠️ Below Target |
| Error Rate | 0.12% | <0.1% | ⚠️ Slightly High |
| Uptime | 99.92% | 99.95% | ⚠️ Close |

### Identified Issues

1. **Cache Inefficiency**: Your Redis cache hit rate is only 67%, indicating:
   - Inefficient cache key strategies
   - Too short TTL values
   - Missing cache warming on deploy

2. **Database Connection Pooling**: I noticed this pattern in your code:

```python
# Current implementation (inefficient)
def get_user_data(user_id):
    conn = psycopg2.connect(DATABASE_URL)  # New connection each time!
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    result = cursor.fetchone()
    conn.close()
    return result

# Recommended implementation
from psycopg2 import pool

class DatabasePool:
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=5,
                maxconn=20,
                dsn=DATABASE_URL
            )
        return cls._instance
    
    def get_connection(self):
        return self._pool.getconn()
    
    def return_connection(self, conn):
        self._pool.putconn(conn)

db_pool = DatabasePool()

def get_user_data(user_id):
    conn = db_pool.get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        return cursor.fetchone()
    finally:
        db_pool.return_connection(conn)
```

3. **API Rate Limiting**: Missing rate limiting leaves you vulnerable to abuse:

```typescript
// Add rate limiting middleware
import rateLimit from 'express-rate-limit';

const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP',
  standardHeaders: true,
  legacyHeaders: false,
});

app.use('/api/', limiter);

// Different limits for different endpoints
const strictLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 10,
  skipSuccessfulRequests: true,
});

app.use('/api/auth/login', strictLimiter);
app.use('/api/auth/register', strictLimiter);
```

### Optimization Recommendations

| Priority | Recommendation | Expected Impact | Effort | Implementation Time |
|----------|---------------|-----------------|--------|---------------------|
| 🔴 High | Implement connection pooling | -30% DB latency | Low | 2 days |
| 🔴 High | Optimize cache strategy | +25% cache hit rate | Medium | 3 days |
| 🟠 Medium | Add API rate limiting | Prevent abuse | Low | 1 day |
| 🟠 Medium | Upgrade to HTTP/2 | -15% latency | Medium | 3 days |
| 🟡 Low | Implement query optimization | -20% query time | High | 1 week |
| 🟡 Low | Add monitoring dashboards | Better visibility | Low | 2 days |

### Proposed Architecture Improvements

Consider implementing these architectural changes:

1. **Add Read Replicas**: Distribute read queries across replicas
2. **Implement CQRS**: Separate read and write models for better scaling
3. **Add Service Mesh**: Better observability and traffic management
4. **Consider Event Sourcing**: For audit trails and debugging

### Cost Impact

Implementing these recommendations would result in:

| Item | Current Cost | Projected Cost | Savings |
|------|--------------|----------------|---------|
| Database Instances | $892/month | $623/month | $269/month |
| API Servers | $547/month | $412/month | $135/month |
| Cache Instances | $125/month | $125/month | $0 |
| **Total** | **$1,564/month** | **$1,160/month** | **$404/month** |

Annual savings: **$4,848**

### Next Steps

1. **Immediate** (This Week):
   - Implement database connection pooling
   - Add basic rate limiting
   - Set up monitoring dashboards

2. **Short-term** (Next Month):
   - Optimize caching strategy
   - Add read replicas
   - Implement comprehensive logging

3. **Long-term** (Next Quarter):
   - Evaluate service mesh options
   - Plan CQRS implementation
   - Consider migration to Kubernetes

Would you like me to provide more detailed implementation guides for any of these recommendations?"""
    
    with StreamingContext(
        console=console,
        title="🏗️ Architecture Analysis",
        mode="response",
        refresh_per_second=10,
        transient=True
    ) as ctx:
        async for chunk in stream_response(mixed_response, speed=0.012):
            ctx.update(chunk)


async def demo_thinking_process():
    """Demonstrate thinking/reasoning process."""
    output = get_output()
    console = Console()
    
    output.print("\n" + "─" * 40)
    output.print("Thinking & Reasoning Process", style="bold cyan")
    output.print("─" * 40)
    output.info("Analyzing complex problem...")
    
    thinking = """Let me think through this optimization problem step by step.

The user wants to optimize a recommendation system that's currently too slow.

Current issues to consider:
- Processing 10M users takes 6 hours
- Memory usage peaks at 32GB
- Recommendations are recalculated from scratch daily
- No caching mechanism in place

Potential approaches:

1. Incremental processing: Instead of recalculating everything, only update changed data
   - Pro: Massive time savings (6 hours → 30 minutes)
   - Con: More complex logic, need change detection

2. Distributed processing: Use Apache Spark or similar
   - Pro: Scales horizontally, handles big data well
   - Con: Infrastructure complexity, learning curve

3. Approximate algorithms: Use locality-sensitive hashing
   - Pro: 100x faster, good enough accuracy
   - Con: Not exact, needs validation

4. Hybrid approach: Combine multiple strategies
   - Real-time for active users
   - Batch for inactive users
   - Cache popular recommendations

Evaluating based on constraints:
- Budget: Limited (rules out expensive infrastructure)
- Team expertise: Python/SQL (rules out complex distributed systems)
- Accuracy requirement: 95% is acceptable (allows approximation)

Best approach: Start with incremental processing + smart caching, then add LSH for similarity calculations.

This should reduce processing time by 85% with minimal infrastructure changes."""
    
    with StreamingContext(
        console=console,
        title="💭 Thinking",
        mode="thinking",
        refresh_per_second=8,
        transient=True
    ) as ctx:
        async for chunk in stream_response(thinking, speed=0.02):
            ctx.update(chunk)


async def demo_json_yaml_config():
    """Demonstrate JSON and YAML configuration examples."""
    output = get_output()
    console = Console()
    
    output.print("\n" + "─" * 40)
    output.print("Configuration & Data Formats", style="bold cyan")
    output.print("─" * 40)
    output.info("Generating configuration examples...")
    
    config_response = """I'll provide configuration examples for your deployment pipeline.

## Docker Compose Configuration

Here's a complete `docker-compose.yml` for your stack:

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    container_name: nginx-proxy
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - static_volume:/usr/share/nginx/html/static
    depends_on:
      - api
    networks:
      - app-network
    restart: unless-stopped

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    container_name: api-server
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/appdb
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET=${JWT_SECRET}
      - S3_BUCKET=${S3_BUCKET}
    ports:
      - "3000:3000"
    depends_on:
      - postgres
      - redis
    networks:
      - app-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  postgres:
    image: postgres:14-alpine
    container_name: postgres-db
    environment:
      - POSTGRES_DB=appdb
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - app-network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: redis-cache
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - app-network
    restart: unless-stopped

  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: background-worker
    environment:
      - NODE_ENV=production
      - DATABASE_URL=postgresql://user:pass@postgres:5432/appdb
      - REDIS_URL=redis://redis:6379
    depends_on:
      - postgres
      - redis
    networks:
      - app-network
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
  static_volume:

networks:
  app-network:
    driver: bridge
```

## Kubernetes Deployment Configuration

And here's the equivalent Kubernetes configuration:

```json
{
  "apiVersion": "apps/v1",
  "kind": "Deployment",
  "metadata": {
    "name": "api-deployment",
    "namespace": "production",
    "labels": {
      "app": "api",
      "version": "v1.0.0",
      "environment": "production"
    }
  },
  "spec": {
    "replicas": 3,
    "selector": {
      "matchLabels": {
        "app": "api"
      }
    },
    "template": {
      "metadata": {
        "labels": {
          "app": "api",
          "version": "v1.0.0"
        }
      },
      "spec": {
        "containers": [
          {
            "name": "api",
            "image": "myregistry.com/api:v1.0.0",
            "ports": [
              {
                "containerPort": 3000,
                "protocol": "TCP"
              }
            ],
            "env": [
              {
                "name": "NODE_ENV",
                "value": "production"
              },
              {
                "name": "DATABASE_URL",
                "valueFrom": {
                  "secretKeyRef": {
                    "name": "db-secret",
                    "key": "url"
                  }
                }
              },
              {
                "name": "REDIS_URL",
                "valueFrom": {
                  "configMapKeyRef": {
                    "name": "app-config",
                    "key": "redis-url"
                  }
                }
              }
            ],
            "resources": {
              "requests": {
                "memory": "256Mi",
                "cpu": "250m"
              },
              "limits": {
                "memory": "512Mi",
                "cpu": "500m"
              }
            },
            "livenessProbe": {
              "httpGet": {
                "path": "/health",
                "port": 3000
              },
              "initialDelaySeconds": 30,
              "periodSeconds": 10
            },
            "readinessProbe": {
              "httpGet": {
                "path": "/ready",
                "port": 3000
              },
              "initialDelaySeconds": 5,
              "periodSeconds": 5
            }
          }
        ],
        "imagePullSecrets": [
          {
            "name": "registry-secret"
          }
        ]
      }
    },
    "strategy": {
      "type": "RollingUpdate",
      "rollingUpdate": {
        "maxSurge": 1,
        "maxUnavailable": 0
      }
    }
  }
}
```

## Application Configuration

Here's a comprehensive `config.json` for your application:

```json
{
  "app": {
    "name": "MyApplication",
    "version": "2.1.0",
    "environment": "production",
    "debug": false,
    "port": 3000,
    "host": "0.0.0.0"
  },
  "database": {
    "primary": {
      "host": "db-primary.example.com",
      "port": 5432,
      "database": "production",
      "user": "app_user",
      "pool": {
        "min": 5,
        "max": 20,
        "idle": 10000
      }
    },
    "replicas": [
      {
        "host": "db-replica-1.example.com",
        "port": 5432,
        "weight": 1
      },
      {
        "host": "db-replica-2.example.com",
        "port": 5432,
        "weight": 2
      }
    ]
  },
  "cache": {
    "redis": {
      "host": "redis.example.com",
      "port": 6379,
      "db": 0,
      "ttl": 3600,
      "prefix": "app:"
    }
  },
  "auth": {
    "jwt": {
      "expiresIn": "24h",
      "refreshExpiresIn": "7d",
      "algorithm": "RS256"
    },
    "oauth": {
      "google": {
        "clientId": "YOUR_GOOGLE_CLIENT_ID",
        "callbackUrl": "https://api.example.com/auth/google/callback"
      },
      "github": {
        "clientId": "YOUR_GITHUB_CLIENT_ID",
        "callbackUrl": "https://api.example.com/auth/github/callback"
      }
    }
  },
  "logging": {
    "level": "info",
    "format": "json",
    "outputs": [
      {
        "type": "console",
        "level": "info"
      },
      {
        "type": "file",
        "path": "/var/log/app/app.log",
        "maxSize": "100m",
        "maxFiles": 10
      },
      {
        "type": "syslog",
        "host": "logs.example.com",
        "port": 514,
        "protocol": "udp"
      }
    ]
  },
  "features": {
    "newDashboard": true,
    "betaFeatures": false,
    "maintenanceMode": false,
    "rateLimiting": {
      "enabled": true,
      "requests": 100,
      "window": "15m"
    }
  }
}
```

These configurations provide a complete setup for containerized deployment with proper health checks, resource limits, and security considerations."""
    
    with StreamingContext(
        console=console,
        title="⚙️ Configuration Examples",
        mode="response",
        refresh_per_second=10,
        transient=True
    ) as ctx:
        async for chunk in stream_response(config_response, speed=0.015):
            ctx.update(chunk)


async def main():
    """Run the full streaming showcase."""
    output = get_output()
    set_theme("default")
    
    # Header
    output.print("\n" + "="*80)
    output.print("MCP-CLI STREAMING SHOWCASE - FULL DEMO", style="bold cyan")
    output.print("Comprehensive demonstration of all streaming capabilities", style="dim")
    output.print("="*80)
    
    # Run all demos
    await demo_code_generation()
    await asyncio.sleep(1)
    
    await demo_tool_execution()
    await asyncio.sleep(1)
    
    await demo_markdown_tables()
    await asyncio.sleep(1)
    
    await demo_thinking_process()
    await asyncio.sleep(1)
    
    await demo_json_yaml_config()
    await asyncio.sleep(1)
    
    await demo_mixed_content()
    
    # Summary
    output.print("\n" + "="*80)
    output.success("✅ Full Streaming Showcase Complete!")
    output.print("="*80)
    
    output.info("\n📊 Demonstrated Capabilities:")
    output.print("• Code generation with syntax highlighting (Python, TypeScript)")
    output.print("• Tool execution with real-time progress")
    output.print("• Complex markdown tables with data")
    output.print("• Thinking/reasoning process display")
    output.print("• JSON/YAML configuration examples")
    output.print("• Mixed content with seamless transitions")
    output.print("• Content-aware phase messages")
    output.print("• Automatic content type detection")
    output.print("• Clean transient display with final panels")
    
    output.print("\n💡 The streaming system automatically:")
    output.print("• Detects content type from early chunks")
    output.print("• Adapts phase messages based on content")
    output.print("• Provides live progress indicators")
    output.print("• Formats final output appropriately")
    output.print("• Handles mixed content seamlessly")


if __name__ == "__main__":
    output = get_output()
    output.print("\n🚀 Starting MCP-CLI Full Streaming Showcase...")
    output.print("This demo shows all content types and streaming capabilities\n")
    asyncio.run(main())