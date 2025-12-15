# mcp_cli/tools/search.py
"""Intelligent tool search with synonym expansion, fuzzy matching, and OR semantics.

This module provides robust tool discovery that matches how LLMs naturally describe
tools they're looking for, rather than requiring exact substring matches.

Key features:
1. Tokenized OR semantics - any matching token scores
2. Synonym expansion - "gaussian" finds "normal", "cdf" finds "cumulative"
3. Fuzzy matching fallback - handles typos and close matches
4. Always returns something - popular tools when nothing else matches
5. Namespace aliasing - "math.normal_cdf" finds "normal_cdf"
6. Two-stage search - high precision first, then expand if needed
7. Session boosting - recently used tools rank higher
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp_cli.tools.models import ToolInfo

logger = logging.getLogger(__name__)


# ============================================================================
# Synonym Dictionary
# ============================================================================

# Bidirectional synonyms - each key maps to related terms
# When searching, we expand the query to include synonyms
SYNONYMS: dict[str, set[str]] = {
    # Statistics / Probability
    "normal": {"gaussian", "bell", "standard"},
    "gaussian": {"normal", "bell", "standard"},
    "cdf": {"cumulative", "distribution", "probability"},
    "cumulative": {"cdf"},
    "pdf": {"probability", "density", "distribution"},
    "mean": {"average", "expected", "mu", "expectation"},
    "average": {"mean", "avg"},
    "std": {"standard", "deviation", "sigma", "stddev"},
    "deviation": {"std", "sigma", "variance"},
    "sigma": {"std", "deviation", "standard"},
    "variance": {"var", "deviation"},
    "median": {"middle", "percentile"},
    "percentile": {"quantile", "quartile"},
    "quantile": {"percentile"},
    "correlation": {"corr", "covariance"},
    "covariance": {"cov", "correlation"},
    "regression": {"linear", "fit", "model"},
    "hypothesis": {"test", "significance", "pvalue"},
    "pvalue": {"significance", "hypothesis", "test"},
    "confidence": {"interval", "ci"},
    "tail": {"probability", "risk"},
    # Math operations
    "add": {"sum", "plus", "addition"},
    "sum": {"add", "total", "aggregate"},
    "subtract": {"minus", "difference", "sub"},
    "multiply": {"times", "product", "mult"},
    "divide": {"division", "quotient", "div"},
    "power": {"exponent", "pow", "exp"},
    "sqrt": {"square", "root"},
    "log": {"logarithm", "ln", "natural"},
    "sin": {"sine", "trig", "trigonometric"},
    "cos": {"cosine", "trig", "trigonometric"},
    "tan": {"tangent", "trig", "trigonometric"},
    "factorial": {"gamma", "permutation"},
    "combination": {"choose", "binomial", "nCr"},
    "permutation": {"nPr", "arrangement"},
    # File operations
    "read": {"get", "load", "fetch", "retrieve", "open"},
    "write": {"save", "store", "put", "create"},
    "delete": {"remove", "rm", "erase", "unlink"},
    "list": {"ls", "dir", "enumerate", "show"},
    "search": {"find", "query", "lookup", "grep"},
    "find": {"search", "locate", "query"},
    # Data types
    "string": {"text", "str", "char"},
    "number": {"int", "integer", "float", "numeric"},
    "array": {"list", "vector", "sequence"},
    "object": {"dict", "map", "hash", "dictionary"},
    "boolean": {"bool", "flag", "true", "false"},
    # Network / API
    "http": {"request", "api", "fetch", "web"},
    "get": {"fetch", "retrieve", "request"},
    "post": {"send", "submit", "create"},
    "json": {"parse", "serialize", "data"},
    "url": {"uri", "link", "endpoint"},
    # Time / Date
    "date": {"time", "datetime", "timestamp"},
    "now": {"current", "today", "present"},
    "format": {"parse", "convert", "transform"},
    # Database
    "query": {"select", "sql", "search"},
    "insert": {"add", "create", "put"},
    "update": {"modify", "change", "set"},
    "database": {"db", "sql", "store"},
}

# ============================================================================
# Domain/Category Detection (for relevance scoring)
# ============================================================================

# Domain indicators - keywords that suggest a tool belongs to a category
# Used to detect domain mismatch and apply penalties
DOMAIN_INDICATORS: dict[str, set[str]] = {
    "statistics": {
        "normal",
        "gaussian",
        "cdf",
        "pdf",
        "probability",
        "distribution",
        "mean",
        "variance",
        "std",
        "deviation",
        "correlation",
        "regression",
        "hypothesis",
        "test",
        "confidence",
        "percentile",
        "quantile",
        "statistical",
        "sample",
        "population",
        "expected",
        "random",
    },
    "number_theory": {
        "prime",
        "collatz",
        "fibonacci",
        "factorial",
        "gcd",
        "lcm",
        "divisor",
        "modulo",
        "congruence",
        "euler",
        "fermat",
        "integer",
        "sequence",
        "series",
        "recursive",
    },
    "arithmetic": {
        "add",
        "subtract",
        "multiply",
        "divide",
        "sum",
        "product",
        "sqrt",
        "power",
        "root",
        "log",
        "exp",
        "abs",
        "round",
    },
    "trigonometry": {
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "radians",
        "degrees",
        "angle",
        "trigonometric",
        "trig",
        "hyperbolic",
    },
    "file_operations": {
        "read",
        "write",
        "file",
        "directory",
        "path",
        "open",
        "close",
        "save",
        "load",
        "delete",
        "create",
        "copy",
        "move",
    },
    "network": {
        "http",
        "request",
        "response",
        "api",
        "url",
        "fetch",
        "post",
        "get",
        "endpoint",
        "server",
        "client",
        "socket",
    },
    "database": {
        "query",
        "sql",
        "select",
        "insert",
        "update",
        "delete",
        "table",
        "database",
        "record",
        "row",
        "column",
        "join",
    },
}

# Query domain detection patterns
QUERY_DOMAIN_PATTERNS: dict[str, set[str]] = {
    "statistics": {
        "risk",
        "probability",
        "stock",
        "inventory",
        "forecast",
        "normal",
        "gaussian",
        "distribution",
        "confidence",
        "interval",
        "hypothesis",
        "significance",
        "variance",
        "deviation",
        "mean",
        "expected",
        "random",
        "sample",
    },
    "number_theory": {
        "prime",
        "collatz",
        "fibonacci",
        "sequence",
        "integer",
        "divisibility",
        "factorization",
    },
}


def detect_query_domain(keywords: list[str]) -> str | None:
    """Detect the likely domain of a query from its keywords.

    Returns the domain name if confidently detected, None otherwise.
    """
    keyword_set = set(k.lower() for k in keywords)

    best_domain = None
    best_score = 0

    for domain, patterns in QUERY_DOMAIN_PATTERNS.items():
        overlap = keyword_set & patterns
        if len(overlap) > best_score:
            best_score = len(overlap)
            best_domain = domain

    # Require at least 1 match to claim a domain
    return best_domain if best_score >= 1 else None


def detect_tool_domain(tool_name: str, description: str | None) -> str | None:
    """Detect the likely domain of a tool from its name and description.

    Returns the domain name if confidently detected, None otherwise.
    """
    # Combine name and description tokens
    text = f"{tool_name} {description or ''}".lower()
    tokens = set(re.split(r"[_\-.\s]+", text))

    best_domain = None
    best_score = 0

    for domain, indicators in DOMAIN_INDICATORS.items():
        overlap = tokens & indicators
        if len(overlap) > best_score:
            best_score = len(overlap)
            best_domain = domain

    # Require at least 1 match to claim a domain
    return best_domain if best_score >= 1 else None


def compute_domain_penalty(
    query_domain: str | None,
    tool_domain: str | None,
) -> float:
    """Compute penalty for domain mismatch.

    Returns a multiplier (1.0 = no penalty, <1.0 = penalized).
    """
    if query_domain is None or tool_domain is None:
        return 1.0  # Can't determine, no penalty

    if query_domain == tool_domain:
        return 1.0  # Same domain, no penalty

    # Different domains - apply penalty
    # Some domains are more "distant" than others
    DOMAIN_DISTANCE: dict[tuple[str, str], float] = {
        # Statistics vs number_theory is a big mismatch
        ("statistics", "number_theory"): 0.3,
        ("number_theory", "statistics"): 0.3,
        # Statistics vs arithmetic is reasonable
        ("statistics", "arithmetic"): 0.8,
        ("arithmetic", "statistics"): 0.8,
    }

    pair = (query_domain, tool_domain)
    return DOMAIN_DISTANCE.get(pair, 0.5)  # Default 50% penalty


# Common stopwords to filter from queries
STOPWORDS: set[str] = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "must",
    "shall",
    "can",
    "need",
    "dare",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "just",
    "and",
    "but",
    "if",
    "or",
    "because",
    "until",
    "while",
    "although",
    "though",
    "even",
    "that",
    "which",
    "who",
    "whom",
    "this",
    "these",
    "those",
    "what",
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "tool",
    "tools",
    "function",
    "functions",
    "help",
    "helps",
    "use",
    "using",
    "used",
    "want",
    "wants",
    "wanted",
    "please",
    "like",
    "something",
    "anything",
    "everything",
    "nothing",
    "calculate",
    "calculation",
    "calculations",
    "computing",
    "compute",
}


# ============================================================================
# Search Result Model
# ============================================================================


@dataclass
class SearchResult:
    """Result of a tool search with scoring information."""

    tool: ToolInfo
    score: float
    match_reasons: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.tool.name

    @property
    def namespace(self) -> str:
        return self.tool.namespace

    @property
    def description(self) -> str | None:
        return self.tool.description


# ============================================================================
# Token Processing
# ============================================================================


def tokenize(text: str) -> list[str]:
    """Tokenize text into searchable terms.

    Handles:
    - snake_case: normal_cdf -> [normal, cdf]
    - camelCase: normalCdf -> [normal, cdf]
    - kebab-case: normal-cdf -> [normal, cdf]
    - dot.notation: math.normal -> [math, normal]
    - Numbers preserved: sin2 -> [sin, 2]
    """
    if not text:
        return []

    # Normalize to lowercase
    text = text.lower()

    # Split on common separators (underscore, dash, dot, space)
    parts = re.split(r"[_\-.\s]+", text)

    tokens = []
    for part in parts:
        if not part:
            continue

        # Split camelCase: normalCdf -> normal, Cdf
        camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", part).lower().split()
        for token in camel_split:
            # Further split on number boundaries: sin2 -> sin, 2
            number_split = re.split(r"(\d+)", token)
            for t in number_split:
                if t and len(t) >= 2:  # Minimum token length
                    tokens.append(t)

    return tokens


def expand_with_synonyms(tokens: list[str]) -> set[str]:
    """Expand tokens with synonyms for broader matching."""
    expanded = set(tokens)

    for token in tokens:
        if token in SYNONYMS:
            expanded.update(SYNONYMS[token])

    return expanded


def extract_keywords(query: str) -> list[str]:
    """Extract meaningful keywords from a natural language query.

    Removes stopwords and extracts the meaningful search terms.
    """
    tokens = tokenize(query)
    keywords = [t for t in tokens if t not in STOPWORDS]

    # If all tokens were stopwords, return original tokens
    return keywords if keywords else tokens


# ============================================================================
# Scoring Functions
# ============================================================================


def score_token_match(
    query_tokens: set[str],
    tool_name: str,
    tool_description: str | None,
    tool_namespace: str,
    param_names: list[str] | None = None,
) -> tuple[float, list[str]]:
    """Score a tool based on token overlap with query.

    Returns (score, match_reasons) tuple.

    Scoring:
    - Name token exact match: 10 points
    - Name token prefix match: 5 points
    - Description token match: 3 points
    - Namespace match: 2 points
    - Parameter name match: 1 point
    """
    score = 0.0
    reasons = []

    # Tokenize tool attributes
    name_tokens = set(tokenize(tool_name))
    desc_tokens = set(tokenize(tool_description or ""))
    ns_tokens = set(tokenize(tool_namespace))
    param_tokens = set()
    if param_names:
        for p in param_names:
            param_tokens.update(tokenize(p))

    # Check each query token
    for qt in query_tokens:
        # Exact name match (highest value)
        if qt in name_tokens:
            score += 10
            reasons.append(f"name:'{qt}'")

        # Prefix match in name (e.g., "norm" matches "normal")
        elif any(nt.startswith(qt) or qt.startswith(nt) for nt in name_tokens):
            score += 5
            reasons.append(f"name_prefix:'{qt}'")

        # Description match
        if qt in desc_tokens:
            score += 3
            reasons.append(f"desc:'{qt}'")

        # Namespace match
        if qt in ns_tokens:
            score += 2
            reasons.append(f"ns:'{qt}'")

        # Parameter name match
        if qt in param_tokens:
            score += 1
            reasons.append(f"param:'{qt}'")

    return score, reasons


def fuzzy_score(query: str, target: str, threshold: float = 0.6) -> float:
    """Calculate fuzzy match score using sequence matching.

    Returns score between 0 and 1, or 0 if below threshold.
    """
    if not query or not target:
        return 0.0

    ratio = SequenceMatcher(None, query.lower(), target.lower()).ratio()
    return ratio if ratio >= threshold else 0.0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


# ============================================================================
# Name Aliasing
# ============================================================================


def normalize_tool_name(name: str) -> set[str]:
    """Generate normalized variants of a tool name for matching.

    Examples:
    - normal_cdf -> {normal_cdf, normalcdf, normal-cdf, normalCdf}
    - math.normal_cdf -> {normal_cdf, math.normal_cdf, ...}
    """
    variants = {name.lower()}

    # Remove namespace prefix if present
    if "." in name:
        base_name = name.split(".")[-1]
        variants.add(base_name.lower())

    # Generate case variants
    base = name.split(".")[-1] if "." in name else name

    # snake_case to other forms
    variants.add(base.lower().replace("_", ""))  # normalcdf
    variants.add(base.lower().replace("_", "-"))  # normal-cdf

    # Convert to camelCase
    parts = base.split("_")
    if len(parts) > 1:
        camel = parts[0].lower() + "".join(p.capitalize() for p in parts[1:])
        variants.add(camel)  # normalCdf

    return variants


def find_tool_by_alias(query_name: str, tools: list[ToolInfo]) -> ToolInfo | None:
    """Find a tool by name, checking aliases and normalized forms."""
    query_variants = normalize_tool_name(query_name)

    for tool in tools:
        tool_variants = normalize_tool_name(tool.name)

        # Check if any query variant matches any tool variant
        if query_variants & tool_variants:
            return tool

        # Also check with namespace prefix
        full_name_variants = normalize_tool_name(f"{tool.namespace}.{tool.name}")
        if query_variants & full_name_variants:
            return tool

    return None


# ============================================================================
# Main Search Function
# ============================================================================


@dataclass
class SessionToolStats:
    """Statistics for a tool's usage in the current session."""

    name: str
    call_count: int = 0
    success_count: int = 0
    last_used_turn: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate of tool calls."""
        return self.success_count / self.call_count if self.call_count > 0 else 0.0


class ToolSearchEngine:
    """Intelligent tool search engine with caching, ranking, and session awareness.

    Features:
    - Two-stage search: high precision first, then expand
    - Session boosting: recently/successfully used tools rank higher
    - Configurable scoring weights
    """

    # Scoring weights (can be tuned)
    WEIGHT_NAME_EXACT = 10.0
    WEIGHT_NAME_PREFIX = 5.0
    WEIGHT_DESC = 3.0
    WEIGHT_NAMESPACE = 2.0
    WEIGHT_PARAM = 1.0

    # Session boost weights
    BOOST_RECENT_USE = 2.0  # Boost for tools used recently
    BOOST_SUCCESS = 1.5  # Additional boost for successful use
    BOOST_CALL_COUNT = 0.5  # Small boost per successful call
    BOOST_DECAY_TURNS = 5  # How many turns before boost decays

    def __init__(self) -> None:
        self._tool_cache: list[ToolInfo] | None = None
        self._search_index: dict[str, set[str]] | None = (
            None  # tool_name -> searchable tokens
        )

        # Session tracking
        self._session_stats: dict[str, SessionToolStats] = {}
        self._current_turn: int = 0

    def set_tools(self, tools: list[ToolInfo]) -> None:
        """Cache tools and build search index."""
        self._tool_cache = tools
        self._build_search_index()

    # =========================================================================
    # Session Tracking
    # =========================================================================

    def record_tool_use(
        self,
        tool_name: str,
        success: bool = True,
    ) -> None:
        """Record a tool usage in the current session.

        This information is used to boost frequently/successfully used tools
        in search results.

        Args:
            tool_name: Name of the tool that was used
            success: Whether the tool call was successful
        """
        if tool_name not in self._session_stats:
            self._session_stats[tool_name] = SessionToolStats(name=tool_name)

        stats = self._session_stats[tool_name]
        stats.call_count += 1
        if success:
            stats.success_count += 1
        stats.last_used_turn = self._current_turn

        logger.debug(
            f"Recorded tool use: {tool_name} (calls={stats.call_count}, "
            f"success_rate={stats.success_rate:.0%})"
        )

    def advance_turn(self) -> None:
        """Advance the session turn counter.

        Call this at the start of each new user prompt to track recency.
        """
        self._current_turn += 1
        logger.debug(f"Advanced to turn {self._current_turn}")

    def reset_session(self) -> None:
        """Reset session statistics (e.g., for a new conversation)."""
        self._session_stats.clear()
        self._current_turn = 0
        logger.debug("Session statistics reset")

    def get_session_boost(self, tool_name: str) -> float:
        """Calculate session-based score boost for a tool.

        Considers:
        - Recency of use (decays over turns)
        - Success rate
        - Call count (with diminishing returns)

        Args:
            tool_name: Name of the tool

        Returns:
            Boost multiplier (1.0 = no boost, >1.0 = boosted)
        """
        if tool_name not in self._session_stats:
            return 1.0

        stats = self._session_stats[tool_name]

        # Base boost for having been used at all
        boost = 1.0

        # Recency boost (decays with turns since last use)
        turns_since_use = self._current_turn - stats.last_used_turn
        if turns_since_use < self.BOOST_DECAY_TURNS:
            recency_factor = 1.0 - (turns_since_use / self.BOOST_DECAY_TURNS)
            boost += self.BOOST_RECENT_USE * recency_factor

        # Success rate boost
        if stats.call_count > 0:
            boost += self.BOOST_SUCCESS * stats.success_rate

        # Call count boost (logarithmic to avoid runaway boosting)
        if stats.success_count > 0:
            import math

            boost += self.BOOST_CALL_COUNT * math.log1p(stats.success_count)

        return boost

    def get_session_stats(self, tool_name: str) -> SessionToolStats | None:
        """Get session statistics for a tool."""
        return self._session_stats.get(tool_name)

    def _build_search_index(self) -> None:
        """Build inverted index for fast searching."""
        if not self._tool_cache:
            self._search_index = {}
            return

        self._search_index = {}
        for tool in self._tool_cache:
            # Collect all searchable tokens for this tool
            tokens = set()

            # Name tokens (with synonyms)
            name_tokens = tokenize(tool.name)
            tokens.update(name_tokens)
            tokens.update(expand_with_synonyms(name_tokens))

            # Description tokens (with synonyms)
            if tool.description:
                desc_tokens = tokenize(tool.description)
                tokens.update(desc_tokens)
                tokens.update(expand_with_synonyms(desc_tokens))

            # Namespace tokens
            tokens.update(tokenize(tool.namespace))

            # Parameter names
            if tool.parameters and "properties" in tool.parameters:
                for param_name in tool.parameters["properties"].keys():
                    tokens.update(tokenize(param_name))

            # Name variants (aliases)
            tokens.update(normalize_tool_name(tool.name))

            self._search_index[tool.name] = tokens

    def search(
        self,
        query: str,
        tools: list[ToolInfo] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
        use_session_boost: bool = True,
    ) -> list[SearchResult]:
        """Search for tools matching the query using two-stage search.

        Stage 1 (High Precision): Exact and prefix token matches only
        Stage 2 (Expanded): Synonym expansion and fuzzy matching

        Session boosting is applied to rank recently/successfully used tools higher.

        Args:
            query: Natural language search query
            tools: Tools to search (uses cache if None)
            limit: Maximum results to return
            min_score: Minimum score threshold (0 = return everything)
            use_session_boost: Whether to apply session-based boosting

        Returns:
            List of SearchResult sorted by score (highest first)
        """
        if tools is not None:
            search_tools = tools
        elif self._tool_cache is not None:
            search_tools = self._tool_cache
        else:
            return []

        if not search_tools:
            return []

        # Extract keywords from query
        keywords = extract_keywords(query)
        if not keywords:
            keywords = tokenize(query)

        if not keywords:
            return self._fallback_results(search_tools, limit, use_session_boost)

        logger.debug(f"Search query='{query}' -> keywords={keywords}")

        # Detect query domain for relevance filtering
        query_domain = detect_query_domain(keywords)
        if query_domain:
            logger.debug(f"Detected query domain: {query_domain}")

        # Stage 1: High precision search (no synonym expansion)
        stage1_results = self._stage1_search(keywords, search_tools, min_score)

        # If Stage 1 found good results, use them
        if stage1_results and stage1_results[0].score >= self.WEIGHT_NAME_EXACT:
            logger.debug(f"Stage 1 found {len(stage1_results)} high-quality results")
            results = stage1_results
        else:
            # Stage 2: Expanded search with synonyms
            query_tokens = expand_with_synonyms(keywords)
            logger.debug(f"Stage 2: expanding to {query_tokens}")
            results = self._stage2_search(
                query_tokens, search_tools, min_score, query_domain
            )

            # If Stage 2 fails, try fuzzy matching
            if not results:
                results = self._fuzzy_search(query, search_tools, limit)

        # If still no results, return fallback
        if not results:
            return self._fallback_results(search_tools, limit, use_session_boost)

        # Apply session boosting
        if use_session_boost:
            results = self._apply_session_boost(results)

        # Sort by final score
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:limit]

    def _stage1_search(
        self,
        keywords: list[str],
        tools: list[ToolInfo],
        min_score: float,
    ) -> list[SearchResult]:
        """Stage 1: High precision search without synonym expansion.

        Only matches exact tokens and prefixes in tool names.
        """
        results: list[SearchResult] = []
        keyword_set = set(keywords)

        for tool in tools:
            name_tokens = set(tokenize(tool.name))
            score = 0.0
            reasons = []

            for kw in keyword_set:
                # Exact name match
                if kw in name_tokens:
                    score += self.WEIGHT_NAME_EXACT
                    reasons.append(f"name:'{kw}'")
                # Prefix match
                elif any(nt.startswith(kw) for nt in name_tokens):
                    score += self.WEIGHT_NAME_PREFIX
                    reasons.append(f"name_prefix:'{kw}'")

            if score > min_score:
                results.append(
                    SearchResult(tool=tool, score=score, match_reasons=reasons)
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _stage2_search(
        self,
        query_tokens: set[str],
        tools: list[ToolInfo],
        min_score: float,
        query_domain: str | None = None,
    ) -> list[SearchResult]:
        """Stage 2: Expanded search with synonym expansion.

        Includes description, namespace, and parameter matching.
        Applies domain penalty for tools from mismatched domains.
        """
        results: list[SearchResult] = []

        for tool in tools:
            param_names = None
            if tool.parameters and "properties" in tool.parameters:
                param_names = list(tool.parameters["properties"].keys())

            score, reasons = score_token_match(
                query_tokens,
                tool.name,
                tool.description,
                tool.namespace,
                param_names,
            )

            # Apply domain penalty for mismatched domains
            if query_domain is not None:
                tool_domain = detect_tool_domain(tool.name, tool.description)
                penalty = compute_domain_penalty(query_domain, tool_domain)
                if penalty < 1.0:
                    score *= penalty
                    reasons.append(f"domain_penalty:{penalty:.1f}x({tool_domain})")

            if score > min_score:
                results.append(
                    SearchResult(tool=tool, score=score, match_reasons=reasons)
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _apply_session_boost(self, results: list[SearchResult]) -> list[SearchResult]:
        """Apply session-based boosting to search results."""
        boosted = []
        for result in results:
            boost = self.get_session_boost(result.tool.name)
            if boost > 1.0:
                result.score *= boost
                result.match_reasons.append(f"session_boost:{boost:.2f}x")
            boosted.append(result)
        return boosted

    def _fuzzy_search(
        self,
        query: str,
        tools: list[ToolInfo],
        limit: int,
    ) -> list[SearchResult]:
        """Fuzzy search as fallback when token matching fails."""
        results: list[SearchResult] = []

        query_lower = query.lower()

        for tool in tools:
            # Check fuzzy match against name
            name_score = fuzzy_score(query_lower, tool.name.lower(), threshold=0.5)

            # Check fuzzy match against description words
            desc_score = 0.0
            if tool.description:
                desc_words = tool.description.lower().split()
                for word in desc_words:
                    word_score = fuzzy_score(query_lower, word, threshold=0.6)
                    if word_score > desc_score:
                        desc_score = word_score

            total_score = (name_score * 10) + (desc_score * 3)

            if total_score > 0:
                reasons = []
                if name_score > 0:
                    reasons.append(f"fuzzy_name:{name_score:.2f}")
                if desc_score > 0:
                    reasons.append(f"fuzzy_desc:{desc_score:.2f}")

                results.append(
                    SearchResult(tool=tool, score=total_score, match_reasons=reasons)
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def _fallback_results(
        self,
        tools: list[ToolInfo],
        limit: int,
        use_session_boost: bool = True,
    ) -> list[SearchResult]:
        """Return fallback results when no matches found.

        Returns tools sorted by:
        1. Session boost (if enabled) - recently used tools first
        2. Tools with shorter names (often more fundamental)
        3. Alphabetically as tiebreaker
        """
        results = [
            SearchResult(tool=t, score=0.1, match_reasons=["fallback"]) for t in tools
        ]

        # Apply session boosting
        if use_session_boost:
            results = self._apply_session_boost(results)

        # Sort by score (with session boost), then by name length, then alphabetically
        results.sort(key=lambda r: (-r.score, len(r.tool.name), r.tool.name))

        return results[:limit]

    def find_exact(
        self,
        name: str,
        tools: list[ToolInfo] | None = None,
    ) -> ToolInfo | None:
        """Find a tool by exact name or alias.

        Checks:
        1. Exact name match
        2. Name with namespace prefix
        3. Normalized aliases (snake_case, camelCase, etc.)
        """
        search_tools = tools if tools is not None else self._tool_cache
        if not search_tools:
            return None

        # Try exact match first
        for tool in search_tools:
            if tool.name == name:
                return tool

        # Try with namespace prefix
        for tool in search_tools:
            if f"{tool.namespace}.{tool.name}" == name:
                return tool

        # Try alias matching
        return find_tool_by_alias(name, search_tools)


# ============================================================================
# Convenience Functions
# ============================================================================

# Global search engine instance
_search_engine: ToolSearchEngine | None = None


def get_search_engine() -> ToolSearchEngine:
    """Get or create the global search engine instance."""
    global _search_engine
    if _search_engine is None:
        _search_engine = ToolSearchEngine()
    return _search_engine


def search_tools(
    query: str,
    tools: list[ToolInfo],
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search for tools matching the query.

    Convenience function that returns results in dict format.
    """
    engine = get_search_engine()
    results = engine.search(query, tools, limit)

    return [
        {
            "name": r.tool.name,
            "description": r.tool.description or "No description",
            "namespace": r.tool.namespace,
            "score": r.score,
            "match_reasons": r.match_reasons,
        }
        for r in results
    ]


def find_tool_exact(
    name: str,
    tools: list[ToolInfo],
) -> ToolInfo | None:
    """Find a tool by exact name or alias."""
    engine = get_search_engine()
    return engine.find_exact(name, tools)
