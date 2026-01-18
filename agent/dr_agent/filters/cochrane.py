"""Cochrane-specific filter for MCP tool results."""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from .base import BaseResultFilter

logger = logging.getLogger(__name__)


def _is_cochrane_url(url: str) -> bool:
    """Check if URL is a Cochrane-related URL."""
    if not url:
        return False
    url_lower = url.lower()
    return "cochrane" in url_lower

def _extract_urls_from_text(text: str) -> List[str]:
    """Extract URLs from text using regex."""
    if not text:
        return []
    url_pattern = r'https?://[^\s\)]+'
    return re.findall(url_pattern, text)

def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string into datetime object.
    
    Supports formats:
    - "23 October 2023"
    - "Oct 23, 2023"
    - "2023-10-23"
    - "Jun 12, 2016"
    - "Mar 8, 2021"
    
    Args:
        date_str: Date string to parse
        
    Returns:
        datetime object or None if parsing fails
    """
    if not date_str:
        return None
    
    # Try common date formats
    date_formats = [
        "%d %B %Y",      # "23 October 2023"
        "%B %d, %Y",     # "October 23, 2023"
        "%b %d, %Y",     # "Oct 23, 2023", "Jun 12, 2016"
        "%d %b %Y",      # "23 Oct 2023"
        "%Y-%m-%d",      # "2023-10-23"
        "%Y-%m",         # "2023-10" (fallback to first day of month)
        "%Y",            # "2023" (fallback to Jan 1)
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    
    # Try parsing with dateutil if available (more flexible)
    try:
        from dateutil import parser
        return parser.parse(date_str)
    except (ImportError, ValueError):
        pass
    
    return None


def create_title_filter_from_list(title_list: Union[List[str], Set[str]]) -> callable:
    """Create a title filter function from a list or set of titles (case-insensitive).
    
    Args:
        title_list: List or set of titles to filter. If a result title contains (case-insensitive)
                    any title from this list/set, it will be filtered. This handles cases where
                    search results have suffixes like " - PubMed" or " - PMC".
                    Matching is done at word boundaries to avoid false matches (e.g., "chronic" 
                    won't match "chronically").
    
    Returns:
        A function that takes a title string and returns True if it should be filtered
    """
    if not title_list:
        return None
    
    # Convert to list of lowercase titles
    title_list_lower = [title.lower().strip() for title in title_list if title]
    
    def title_filter(title: str) -> bool:
        title_lower = title.lower()
        
        # Remove truncation markers (e.g., "...", trailing dots/spaces)
        # Handle both "..." and trailing dots - be more aggressive
        title_clean = title_lower.replace('...', ' ').replace('…', ' ').strip()  # Replace with space to preserve word boundaries
        # Remove trailing dots and ellipsis patterns
        title_clean = re.sub(r'\.{2,}\s*$', '', title_clean).strip()
        title_clean = title_clean.rstrip('. ').strip()
        # Normalize multiple spaces
        title_clean = re.sub(r'\s+', ' ', title_clean).strip()
        # Normalize hyphens (replace Unicode hyphens with regular hyphens for matching)
        # This handles cases where titles use different hyphen characters (‐, −, –, —, etc.)
        title_clean = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015-]', '-', title_clean)
        
        # Remove common prefixes like [PDF], [Review], [Article], etc.
        # Match patterns like [pdf], [review], [article], etc. (case-insensitive)
        title_clean = re.sub(r'^\[[^\]]+\]\s*', '', title_clean).strip()
        
        # Remove "Web Annex" prefixes (e.g., "Web Annex B.8.", "Web Annex 4.12.")
        # Pattern matches: "Web Annex" followed by alphanumeric/dots (like "B.8", "4.12") and optional period
        # More flexible pattern to catch variations
        title_clean = re.sub(r'^web\s+annex\s+[a-z0-9.]+\.?\s*', '', title_clean, flags=re.IGNORECASE).strip()
        # Also try a more general pattern in case the first one doesn't match
        if title_clean.startswith('web annex'):
            # Remove everything up to and including the first period or space after "annex"
            title_clean = re.sub(r'^web\s+annex[^a-z]*', '', title_clean, flags=re.IGNORECASE).strip()
        
        # Remove common suffixes like " - PubMed", " - PMC", " | PubMed", etc.
        # Also handle patterns like " - ...", " | ...", etc.
        title_clean = re.sub(r'\s*[-|]\s*(PubMed|PMC|NCBI|pubmed|pmc|ncbi|\.\.\.|…).*$', '', title_clean).strip()
        
        # Remove trailing ellipsis or dots that might remain
        title_clean = title_clean.rstrip('.… ').strip()
        
        for list_title in title_list_lower:
            list_title_clean = list_title.strip()
            # Remove common prefixes from filter title too (for consistency)
            list_title_clean = re.sub(r'^\[[^\]]+\]\s*', '', list_title_clean).strip()
            # Remove "Web Annex" prefixes from filter title too (for consistency)
            list_title_clean = re.sub(r'^web\s+annex\s+[a-z0-9.]+\.?\s*', '', list_title_clean, flags=re.IGNORECASE).strip()
            # Also try a more general pattern in case the first one doesn't match
            if list_title_clean.startswith('web annex'):
                list_title_clean = re.sub(r'^web\s+annex[^a-z]*', '', list_title_clean, flags=re.IGNORECASE).strip()
            # Remove suffixes from filter title
            list_title_clean = re.sub(r'\s*[-|]\s*(PubMed|PMC|NCBI|pubmed|pmc|ncbi|\.\.\.|…).*$', '', list_title_clean).strip()
            list_title_clean = list_title_clean.rstrip('.… ').strip()
            # Normalize hyphens (replace Unicode hyphens with regular hyphens for matching)
            # This handles cases where titles use different hyphen characters (‐, −, –, —, etc.)
            list_title_clean = re.sub(r'[\u2010\u2011\u2012\u2013\u2014\u2015-]', '-', list_title_clean)
            
            if not title_clean or not list_title_clean:
                continue
            
            # Method 1: Exact substring match (handles suffixes like " - PubMed" that weren't caught)
            # Check if list title appears as a complete phrase in cleaned result title
            # Require at least 4 words to avoid false positives with short titles
            list_title_words = list_title_clean.split()
            if len(list_title_words) >= 4 and list_title_clean in title_clean:
                # Verify it's at word boundaries (not part of another word)
                escaped = re.escape(list_title_clean)
                # Match if it's at word boundaries or is the exact string
                if re.search(rf'\b{escaped}\b|^{escaped}$', title_clean):
                    return True
            
            # Method 1b: Check if result title appears in list title (reverse direction)
            # This handles truncated search result titles - check if truncated title matches start of full title
            # This is the PRIMARY method for truncated titles like "Parallel Use of Low-Complexity Automated Nucleic Acid ..."
            title_words = title_clean.split()
            if len(title_words) >= 3:
                # PRIMARY CHECK: Check if ALL words in the filtered title match the start of the Cochrane title
                # This is the most accurate check - compare all available words, not just first N
                if list_title_clean.startswith(title_clean):
                    return True
                
                # SECONDARY CHECK: Only use progressively shorter prefixes if we suspect truncation
                # Check if title appears truncated (ends with ellipsis, is unusually short, etc.)
                # Otherwise, require ALL words to match
                appears_truncated = (
                    title_clean.endswith('...') or 
                    title_clean.endswith('…') or
                    len(title_words) < 8  # Titles with fewer than 8 words might be truncated
                )
                
                if appears_truncated:
                    # For truncated titles, check progressively shorter prefixes
                    # Check 7, 6 words to catch different truncation points
                    for prefix_len in [7, 6]:
                        if len(title_words) >= prefix_len:
                            title_prefix = ' '.join(title_words[:prefix_len])
                            if list_title_clean.startswith(title_prefix):
                                # Additional validation: if we have more words, check that the next word also matches
                                # This prevents false positives where titles share a common prefix but diverge
                                if len(title_words) > prefix_len:
                                    # Get the next word after the prefix
                                    next_word = title_words[prefix_len] if prefix_len < len(title_words) else None
                                    if next_word:
                                        # Check if the Cochrane title has the same next word
                                        list_title_words = list_title_clean.split()
                                        if len(list_title_words) > prefix_len:
                                            list_next_word = list_title_words[prefix_len]
                                            if next_word.lower() == list_next_word.lower():
                                                return True
                                        # If next word doesn't match, don't consider it a match (likely different papers)
                                else:
                                    # If the filtered title is exactly this length, it's a match
                                    return True
                # If title doesn't appear truncated, we already checked full match above, so no need for prefix fallback
            
            # Method 1c: Check if significant portion of result title matches start of list title
            # This is a fallback for cases where truncation happens mid-word or with slight variations
            # Only use this if the title appears truncated (otherwise Method 1b should have caught it)
            appears_truncated = (
                title_clean.endswith('...') or 
                title_clean.endswith('…') or
                len(title_words) < 8  # Titles with fewer than 8 words might be truncated
            )
            
            if appears_truncated and len(title_words) >= 6:
                # Take first 6 words of result title
                title_prefix = ' '.join(title_words[:6])
                if list_title_clean.startswith(title_prefix):
                    # Additional validation: check that the next word also matches (if available)
                    # This prevents false positives where titles share a common prefix but diverge
                    if len(title_words) > 6:
                        next_word = title_words[6]
                        list_title_words = list_title_clean.split()
                        if len(list_title_words) > 6:
                            list_next_word = list_title_words[6]
                            if next_word.lower() != list_next_word.lower():
                                # Next word doesn't match - likely different papers, don't filter
                                pass
                            else:
                                # Next word matches - ensure it's at word boundaries
                                escaped_prefix = re.escape(title_prefix)
                                if re.search(rf'^{escaped_prefix}\b', list_title_clean):
                                    return True
                    else:
                        # Filtered title is exactly 6 words - check word boundaries
                        escaped_prefix = re.escape(title_prefix)
                        if re.search(rf'^{escaped_prefix}\b', list_title_clean):
                            return True
            
            # Method 2: Check if cleaned result title appears at the beginning of list title (handles truncation)
            # Require at least 3 words (reduced from 4) to catch more truncated titles
            title_words = title_clean.split()
            if len(title_words) >= 3 and list_title_clean.startswith(title_clean):
                # Additional check: ensure the next character (if any) is a word boundary
                if len(list_title_clean) == len(title_clean) or not list_title_clean[len(title_clean):len(title_clean)+1].isalnum():
                    return True
            
            # Method 3: List title starts with cleaned result title (handles truncation at end)
            # Require at least 3 words (reduced from 4) to catch more truncated titles
            if len(title_clean.split()) >= 3 and list_title_clean.startswith(title_clean):
                # Additional check: ensure the next character (if any) is a word boundary
                if len(list_title_clean) == len(title_clean) or not list_title_clean[len(title_clean):len(title_clean)+1].isalnum():
                    return True
        
        return False
    
    return title_filter


class CochraneResultFilter(BaseResultFilter):
    """Filter that removes Cochrane-related results from search tools.
    
    This filter applies to:
    - serper_google_webpage_search: Filters organic search results
    - semantic_scholar_snippet_search: Filters semantic scholar snippets
    - jina_fetch_webpage_content: Filters webpage content based on Cochrane mentions and titles
    
    Filters out items with:
    - URLs matching Cochrane domains (cochranelibrary.com, cochrane.org, etc.)
    - Titles containing "Cochrane" (case-insensitive)
    - Titles matching optional custom title filter
    - Jina API content that contains "Cochrane" AND the source_title
    
    Example:
        # Basic usage (filters Cochrane URLs only)
        filter = CochraneResultFilter()
        
        # With title filter list (filters titles that exactly match any title in the list)
        filter = CochraneResultFilter(title_filter_list=["Title 1", "Title 2", "Title 3"])
    """
    
    def __init__(
        self, 
        title_filter_list: Optional[Union[List[str], Set[str]]] = None,
        source_title: Optional[str] = None,
        publication_date: Optional[str] = None
    ):
        """Initialize the filter.
        
        Args:
            title_filter_list: Optional list or set of titles. If provided, creates a filter that
                              filters titles that exactly match (case-insensitive) any title in
                              this list/set. Used for search result filtering.
            source_title: Optional specific source title to filter. Used for Jina API content filtering.
                         If not provided but title_filter_list has one item, uses that as source_title.
            publication_date: Optional date string (e.g., "23 October 2023"). Items published
                             after this date will be filtered out. Supports various date formats.
        """
        self.title_filter_list = title_filter_list if title_filter_list else None
        
        if title_filter_list:
            self.title_filter = create_title_filter_from_list(title_filter_list)
        else:
            self.title_filter = None
        
        # Set source_title for Jina content filtering
        # Use provided source_title, or if title_filter_list has exactly one item, use that
        if source_title:
            self.source_title = source_title
        elif title_filter_list and len(title_filter_list) == 1:
            self.source_title = list(title_filter_list)[0]
        else:
            self.source_title = None
        
        # Parse publication date
        self.publication_date_cutoff = None
        if publication_date:
            parsed_date = _parse_date(publication_date)
            if parsed_date:
                self.publication_date_cutoff = parsed_date
            else:
                logger.warning("Could not parse publication_date: %s", publication_date)
        
        # Tools that this filter applies to
        self.filtered_tools = {
            "serper_google_webpage_search",
            "semantic_scholar_snippet_search",
            "jina_fetch_webpage_content"
        }
        
        # Track unique filtered links throughout the tool calling process
        self.filtered_links: Set[str] = set()
    
    def should_filter_tool(self, tool_name: str) -> bool:
        """Check if this filter should be applied to the given tool.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if the tool should be filtered, False otherwise
        """
        return tool_name in self.filtered_tools
    
    def _should_filter_item(
        self, 
        title: str, 
        urls: List[str], 
        publication_date: Optional[str] = None
    ) -> tuple[bool, Optional[str]]:
        """Determine if an item should be filtered and return reason.
        
        Args:
            title: Item title to check
            urls: List of URLs associated with the item
            publication_date: Optional publication date string to check
            
        Returns:
            Tuple of (should_filter, reason)
        """
        # FIRST: Check if any URL has been filtered before (most robust check)
        # This handles cases where the same URL appears with different titles
        try:
            from ..utils.utils import is_url_filtered
            for url in urls:
                if url and is_url_filtered(url):
                    return True, f"URL was previously filtered: {url}"
        except ImportError:
            # Try remote_mcp_servers utils as fallback (for remote MCP servers)
            try:
                from data_querying.remote_mcp_servers.utils import is_url_filtered
                for url in urls:
                    if url and is_url_filtered(url):
                        return True, f"URL was previously filtered: {url}"
            except ImportError:
                # If utils module not available, skip this check
                pass
        
        # Check URLs for Cochrane
        for url in urls:
            if _is_cochrane_url(url):
                return True, f"Cochrane URL: {url}"
        
        # Check if title contains "Cochrane" (case-insensitive)
        if title and "cochrane" in title.lower():
            return True, f"Title contains 'Cochrane': {title}"
        
        # Check title with custom filter if provided
        if self.title_filter and title:
            try:
                if self.title_filter(title):
                    logger.debug(f"🔍 [FILTER] Title filter matched for: {title[:100]}")
                    return True, f"Title filter matched: {title[:100]}"
            except Exception as e:
                logger.warning(f"⚠️ [FILTER] Error in title filter for '{title[:50]}...': {e}")
                # If filter fails, don't filter the item (fail open)
        
        # Check publication date if cutoff is set
        # Only filter if the item has a date - items without dates are kept
        if self.publication_date_cutoff and publication_date:
            item_date = _parse_date(publication_date)
            if item_date and item_date > self.publication_date_cutoff:
                return True, f"Published after cutoff date ({self.publication_date_cutoff.strftime('%d %B %Y')}): {publication_date}"
            # If date exists but couldn't be parsed, log a warning but don't filter
            elif publication_date and not item_date:
                logger.debug("Could not parse date '%s' for item '%s'", publication_date, title)
        
        return False, None
    
    def _should_filter_jina_content(
        self,
        content: str,
        title: str,
        url: str
    ) -> tuple[bool, Optional[str]]:
        """Determine if Jina API content should be filtered.
        
        Filters content if BOTH conditions are met:
        1. Content contains "Cochrane" (case-insensitive)
        2. AND content contains the source_title
        
        If no source_title is provided, content is not filtered (both conditions required).
        
        Args:
            content: The webpage content text to check
            title: The webpage title
            url: The webpage URL
            
        Returns:
            Tuple of (should_filter, reason)
        """
        if not content:
            return False, None
        
        # Both conditions required - if no source title, can't filter
        if not self.source_title:
            return False, None
        
        content_lower = content.lower()
        source_title_lower = self.source_title.lower()
        
        # Check if content contains "Cochrane"
        contains_cochrane = "cochrane" in content_lower
        
        if not contains_cochrane:
            return False, None
        
        # Check if content contains the source title
        # Check if the title (or a substantial portion) appears in content
        if source_title_lower in content_lower:
            return True, f"Content contains 'Cochrane' and source title: {self.source_title}"
        # Contains "Cochrane" but source title not found - don't filter
        # (might be a general mention of Cochrane, not the specific review)
        return False, None
    
    def _filter_list_items(
        self,
        items: List[Dict[str, Any]],
        get_title: callable,
        get_urls: callable,
        get_metadata: callable,
        get_date: callable,
        tool_name: str = ""
    ) -> List[Dict[str, Any]]:
        """Filter a list of items based on title, URL, and date criteria.
        
        Args:
            items: List of items to filter
            get_title: Function(item) -> str to extract title
            get_urls: Function(item) -> List[str] to extract URLs
            get_metadata: Function(item) -> Dict for logging metadata
            get_date: Function(item) -> Optional[str] to extract publication date
            tool_name: Name of tool for logging
            
        Returns:
            Filtered list of items
        """
        filtered_items = []
        original_count = len(items)
        items_filtered_out = 0
        
        # Iterate through items in order and preserve order in filtered list
        for item in items:
            title = get_title(item)
            urls = get_urls(item)
            date = get_date(item)
            should_filter, reason = self._should_filter_item(title, urls, date)
            
            if should_filter:
                items_filtered_out += 1
                metadata = get_metadata(item)
                logger.info("  [FILTERED] %s", reason)
                logger.info("    Title: %s", title)
                if date:
                    logger.info("    Date:  %s", date)
                for key, value in metadata.items():
                    if value:
                        logger.info("    %s:   %s", key, value)
                
                # Track filtered URLs (both local and global)
                for url in urls:
                    if url:  # Only track non-empty URLs
                        self.filtered_links.add(url)
                        # Also track in global set for cross-tool filtering
                        try:
                            from ..utils.utils import track_filtered_urls
                            track_filtered_urls([url])
                        except ImportError:
                            # If utils module not available, skip global tracking
                            pass
            else:
                # DEBUG: Log items that passed the filter to help diagnose why they weren't filtered
                if self.title_filter and title:
                    # Check if title filter would match (for debugging)
                    try:
                        title_matches = self.title_filter(title)
                        if title_matches:
                            logger.warning(f"⚠️ [FILTER] Title filter matched but item was NOT filtered! Title: {title[:100]}")
                            logger.warning(f"   This indicates a bug in _should_filter_item logic")
                            logger.warning(f"   URLs: {urls}")
                        else:
                            logger.debug(f"✅ [FILTER] Item passed filter check - Title: {title[:100]}")
                    except Exception as e:
                        logger.warning(f"⚠️ [FILTER] Error checking title filter for '{title[:50]}...': {e}")
                elif not self.title_filter:
                    # Log if title filter is not configured
                    logger.debug(f"ℹ️ [FILTER] Title filter not configured - Title: {title[:100]}")
                # Append items in original order (no reordering)
                filtered_items.append(item)
        
        # Update position indices for filtered items
        for idx, item in enumerate(filtered_items, start=1):
            if "position" in item:
                item["position"] = idx
        
        remaining_count = len(filtered_items)
        if original_count != remaining_count:
            logger.info("-" * 80)
            logger.info("  Filtered: %d out of %d results (%d remaining)", 
                      items_filtered_out, original_count, remaining_count)
        
        # Warn if all results were filtered
        if original_count > 0 and remaining_count == 0:
            logger.warning("  WARNING: All %d results were filtered out", original_count)
        
        logger.info("=" * 80)
        
        return filtered_items
    
    def filter(self, tool_result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Filter search results from supported tools.
        
        Supported tools:
        - serper_google_webpage_search: Filters organic search results
        - semantic_scholar_snippet_search: Filters semantic scholar snippets
        - jina_fetch_webpage_content: Filters webpage content based on Cochrane mentions and titles
        
        Args:
            tool_result: The tool result dictionary to filter
            tool_name: Name of the tool that produced the result
            
        Returns:
            Filtered tool result dictionary
        """
        # Log filter configuration (matching sciconbench_code style)
        logger.info("=" * 80)
        logger.info("FILTERING: %s", tool_name)
        logger.info("-" * 80)
        if self.title_filter_list:
            logger.info("  Title filter: %d titles loaded", len(self.title_filter_list))
        if self.source_title:
            logger.info("  Source title: %s", self.source_title[:80])
        if self.publication_date_cutoff:
            logger.info("  Date cutoff: %s", self.publication_date_cutoff.strftime('%d %B %Y'))
        logger.info("-" * 80)
        
        filtered_result = tool_result.copy()
        
        if tool_name == "jina_fetch_webpage_content":
            # Jina returns a single result, not a list
            if isinstance(filtered_result, dict) and filtered_result.get("success", False):
                content = filtered_result.get("content", "")
                title = filtered_result.get("title", "")
                url = filtered_result.get("url", "")
                
                should_filter, reason = self._should_filter_jina_content(content, title, url)
                
                if should_filter:
                    logger.info("=" * 80)
                    logger.info("FILTERING: jina_fetch_webpage_content")
                    logger.info("-" * 80)
                    logger.info("  [FILTERED] %s", reason)
                    logger.info("    Title: %s", title)
                    logger.info("    URL:   %s", url)
                    logger.info("=" * 80)
                    
                    # Track filtered URL (both local and global)
                    if url:  # Only track non-empty URLs
                        self.filtered_links.add(url)
                        # Also track in global set for cross-tool filtering
                        try:
                            from ..utils.utils import track_filtered_urls
                            track_filtered_urls([url])
                        except ImportError:
                            # If utils module not available, skip global tracking
                            pass
                    
                    # Return result with empty content (success=True but content is empty)
                    return {
                        "url": url,
                        "title": title,
                        "content": "",
                        "description": "",
                        "publishedTime": filtered_result.get("publishedTime", ""),
                        "metadata": filtered_result.get("metadata", {}),
                        "success": True,
                    }
        
        elif tool_name == "serper_google_webpage_search":
            if "organic" in filtered_result and isinstance(filtered_result["organic"], list):
                def get_title(item): return item.get("title", "")
                def get_urls(item): return [item.get("link", "")]
                def get_metadata(item): return {"Link": item.get("link", "")}
                def get_date(item): return item.get("date")
                
                filtered_result["organic"] = self._filter_list_items(
                    filtered_result["organic"],
                    get_title, get_urls, get_metadata, get_date,
                    "search"
                )
        
        elif tool_name == "semantic_scholar_snippet_search":
            if "data" in filtered_result and isinstance(filtered_result["data"], list):
                def get_title(item):
                    return item.get("paper", {}).get("title", "")
                
                def get_urls(item):
                    """Extract URLs from Semantic Scholar API result item.
                    
                    Checks:
                    1. openAccessInfo.disclaimer for URLs (PMC, DOI, etc.)
                    2. paper.url field
                    3. Constructs from paperId or corpusId if available
                    """
                    urls = []
                    paper = item.get("paper", {})
                    
                    # First, try to extract URL from disclaimer
                    open_access_info = paper.get("openAccessInfo", {})
                    disclaimer_text = open_access_info.get("disclaimer", "")
                    
                    if disclaimer_text:
                        import re
                        url_pattern = r'https?://[^\s,)]+'
                        found_urls = re.findall(url_pattern, disclaimer_text)
                        
                        if found_urls:
                            # Prefer PMC URLs, then DOI URLs, then other URLs (skip unpaywall API URLs)
                            for found_url in found_urls:
                                if 'pmc.ncbi.nlm.nih.gov' in found_url:
                                    urls.append(found_url)
                                    break
                                elif 'doi.org' in found_url and 'unpaywall.org' not in found_url:
                                    urls.append(found_url)
                                    break
                                elif 'unpaywall.org' not in found_url:
                                    urls.append(found_url)
                                    break
                            
                            # If no preferred URL found, use first non-unpaywall URL
                            if not urls:
                                for found_url in found_urls:
                                    if 'unpaywall.org' not in found_url:
                                        urls.append(found_url)
                                        break
                    
                    # Second, try direct URL from paper
                    if not urls:
                        paper_url = paper.get("url", "")
                        if paper_url:
                            urls.append(paper_url)
                    
                    # Third, construct from paperId or corpusId
                    if not urls:
                        paper_id = paper.get("paperId")
                        corpus_id = paper.get("corpusId")
                        if paper_id:
                            urls.append(f"https://www.semanticscholar.org/paper/{paper_id}")
                        elif corpus_id:
                            urls.append(f"https://www.semanticscholar.org/paper/{corpus_id}")
                    
                    return urls
                
                def get_metadata(item):
                    paper = item.get("paper", {})
                    return {"Corpus ID": paper.get("corpusId", "")}
                
                # No date filtering for semantic scholar - use None as get_date
                def get_date(item):
                    return None
                
                filtered_result["data"] = self._filter_list_items(
                    filtered_result["data"],
                    get_title, get_urls, get_metadata, get_date,
                    "semantic scholar snippet"
                )
        
        # Log summary of filtered links
        filtered_links_count = len(self.filtered_links)
        if filtered_links_count > 0:
            logger.info("  Total filtered URLs tracked: %d", filtered_links_count)
        return filtered_result
    
    def get_filtered_links(self) -> List[str]:
        """Get list of unique filtered links.
        
        Returns:
            Sorted list of unique filtered URLs
        """
        return sorted(list(self.filtered_links))
    
    def reset_filtered_links(self):
        """Reset the filtered links tracking (useful for testing or reusing filter instances)."""
        self.filtered_links.clear()


# Convenience: default Cochrane filter instance (URLs only, no title filter)
custom_cochrane_filter_search_results = CochraneResultFilter()

