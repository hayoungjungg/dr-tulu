import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import dotenv
import requests
from requests.exceptions import ReadTimeout, RequestException, Timeout, ConnectionError, ConnectTimeout
from typing_extensions import TypedDict

from ..cache import cached

logger = logging.getLogger(__name__)

# Try to load .env from multiple possible locations
env_paths = [
    ".env",  # Current directory
    Path(__file__).resolve().parent.parent.parent.parent.parent / ".env",  # cochrane-benchmark/.env
]
for env_path in env_paths:
    if Path(env_path).exists():
        dotenv.load_dotenv(env_path)
        break
else:
    # Fallback: try loading from current directory anyway
    dotenv.load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
TIMEOUT = int(os.getenv("API_TIMEOUT", 10))


class KnowledgeGraph(TypedDict, total=False):
    title: str
    type: str
    website: str
    imageUrl: str
    description: str
    descriptionSource: str
    descriptionLink: str
    attributes: Optional[Dict[str, str]]


class Sitelink(TypedDict):
    title: str
    link: str


class SearchResult(TypedDict):
    title: str
    link: str
    snippet: str
    position: int
    sitelinks: Optional[List[Sitelink]]
    attributes: Optional[Dict[str, str]]
    date: Optional[str]


class PeopleAlsoAsk(TypedDict):
    question: str
    snippet: str
    title: str
    link: str


class RelatedSearch(TypedDict):
    query: str


class SearchResponse(TypedDict, total=False):
    searchParameters: Dict[str, Union[str, int, bool]]
    knowledgeGraph: Optional[KnowledgeGraph]
    organic: List[SearchResult]
    peopleAlsoAsk: Optional[List[PeopleAlsoAsk]]
    relatedSearches: Optional[List[RelatedSearch]]


class ScholarResult(TypedDict):
    title: str
    link: str
    publicationInfo: str
    snippet: str
    year: Union[int, str]
    citedBy: int


class ScholarResponse(TypedDict):
    searchParameters: Dict[str, Union[str, int, bool]]
    organic: List[ScholarResult]


class WebpageContentResponse(TypedDict, total=False):
    url: str
    text: str
    markdown: str
    metadata: Dict[str, Union[str, int, bool]]
    credits: int


@cached()
def search_serper(
    query: str,
    num_results: int = 10,
    gl: str = "us",
    hl: str = "en",
    search_type: str = "search",  # Can be "search", "places", "news", "images"
    page: Optional[int] = None,  # Page number (1-indexed). If None or 1, fetches page 1.
    api_key: str = None,
    max_retries: int = 3,
    timeout: int = TIMEOUT,
) -> SearchResponse:
    """
    Search using Serper.dev API for general web search.
    Fetches a single page of results. If page is None or 1, fetches page 1 (no page parameter).
    If page > 1, includes the page parameter in the request.

    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)
        gl: Country code to boosts search results whose country of origin matches the parameter value (default: us)
        hl: Host language of user interface (default: en)
        search_type: Type of search to perform (default: "search")
                    Options: "search", "places", "news", "images"
        page: Optional page number (1-indexed). If None or 1, fetches page 1 without page parameter.
        api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)
        max_retries: Maximum number of retry attempts for retryable errors (default: 3)
        timeout: Request timeout in seconds (default: TIMEOUT)

    Returns:
        SearchResponse containing:
        - searchParameters: Dict with search metadata
        - knowledgeGraph: Optional knowledge graph information
        - organic: List of organic search results
        - peopleAlsoAsk: Optional list of related questions
        - relatedSearches: Optional list of related search queries
    """
    if not api_key:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "SERPER_API_KEY environment variable is not set or api_key parameter not provided"
            )

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    # Build payload: page 1 (or None) doesn't include page parameter, page > 1 does
    payload_dict = {"q": query, "num": num_results, "gl": gl, "hl": hl, "type": search_type}
    if page is not None and page > 1:
        payload_dict["page"] = page
    
    payload = json.dumps(payload_dict)
    page_num = page if page is not None else 1
    logger.debug(f"Serper API request (page {page_num}): query='{query[:50]}...', gl={gl}, hl={hl}")

    # Retry loop
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=timeout)
            
            # Handle non-200 status codes
            if response.status_code != 200:
                # Retry on server errors (500+) and gateway errors (502, 503, 504)
                retryable_status_codes = {502, 503, 504}
                is_server_error = response.status_code >= 500
                is_retryable = is_server_error or response.status_code in retryable_status_codes
                
                if is_retryable and attempt < max_retries - 1:
                    last_error = f"API error {response.status_code}: {response.text[:200]}"
                    wait_time = random.uniform(10, 15)
                    logger.warning("Serper API error %d. Waiting %.1f seconds before retry %d/%d...", 
                                response.status_code, wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)
                    continue
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text[:500]}"
                )
            
            result = response.json()
            organic = result.get("organic", [])
            logger.debug(f"Serper API returned {len(organic)} results for page {page_num}")
            
            return {
                "searchParameters": result.get("searchParameters", {}),
                "organic": organic,
                "knowledgeGraph": result.get("knowledgeGraph"),
                "peopleAlsoAsk": result.get("peopleAlsoAsk", []),
                "relatedSearches": result.get("relatedSearches", []),
            }
            
        except (ReadTimeout, Timeout, ConnectTimeout) as e:
            last_error = f"Request timed out after {timeout} seconds"
            if attempt < max_retries - 1:
                wait_time = random.uniform(10, 15)
                logger.warning("Serper API timeout error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise Exception(f"{last_error}: {str(e)}")
        except ConnectionError as e:
            # Connection errors (network issues, upstream errors) should be retried
            last_error = f"Connection error: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = random.uniform(10, 15)
                logger.warning("Serper API connection error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise Exception(f"Error performing Serper search: {last_error}")
        except RequestException as e:
            # Other request exceptions should also be retried
            last_error = f"Request failed: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = random.uniform(10, 15)
                logger.warning("Serper API request error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise Exception(f"Error performing Serper search: {last_error}")

    raise Exception(f"Serper API request failed after {max_retries} attempts: {last_error or 'Unknown error'}")


@cached()
def search_serper_scholar(
    query: str,
    num_results: int = 10,
    api_key: str = None,
    max_retries: int = 3,
    timeout: int = TIMEOUT,
) -> ScholarResponse:
    """
    Search academic papers using Serper.dev Scholar API.

    Args:
        query: Academic search query string
        num_results: Number of results to return (default: 10)
        api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)
        max_retries: Maximum number of retry attempts for retryable errors (default: 3)
        timeout: Request timeout in seconds (default: TIMEOUT)

    Returns:
        ScholarResponse containing:
        - organic: List of academic paper results with:
            - title: Paper title
            - link: URL to the paper
            - publicationInfo: Author and publication details
            - snippet: Brief excerpt from the paper
            - year: Publication year
            - citedBy: Number of citations
    """
    if not api_key:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "SERPER_API_KEY environment variable is not set or api_key parameter not provided"
            )

    url = "https://google.serper.dev/scholar"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    # Retry loop
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=timeout)

            # Handle non-200 status codes
            if response.status_code != 200:
                # Retry on server errors (500+) and gateway errors (502, 503, 504)
                retryable_status_codes = {502, 503, 504}
                is_server_error = response.status_code >= 500
                is_retryable = is_server_error or response.status_code in retryable_status_codes
                
                if is_retryable and attempt < max_retries - 1:
                    last_error = f"API error {response.status_code}: {response.text[:200]}"
                    wait_time = random.uniform(10, 15)
                    logger.warning("Serper Scholar API error %d. Waiting %.1f seconds before retry %d/%d...", 
                                response.status_code, wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)
                    continue
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text[:500]}"
                )

            return response.json()

        except (ReadTimeout, Timeout, ConnectTimeout) as e:
            last_error = f"Request timed out after {timeout} seconds"
            if attempt < max_retries - 1:
                wait_time = random.uniform(10, 15)
                logger.warning("Serper Scholar API timeout error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise Exception(f"{last_error}: {str(e)}")
        except ConnectionError as e:
            # Connection errors (network issues, upstream errors) should be retried
            last_error = f"Connection error: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = random.uniform(10, 15)
                logger.warning("Serper Scholar API connection error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise Exception(f"Error performing Serper scholar search: {last_error}")
        except RequestException as e:
            # Other request exceptions should also be retried
            last_error = f"Request failed: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = random.uniform(10, 15)
                logger.warning("Serper Scholar API request error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise Exception(f"Error performing Serper scholar search: {last_error}")

    raise Exception(f"Serper Scholar API request failed after {max_retries} attempts: {last_error or 'Unknown error'}")


@cached()
def fetch_webpage_content(
    url: str,
    include_markdown: bool = True,
    api_key: str = None,
    max_retries: int = 3,
    timeout: int = TIMEOUT,
) -> WebpageContentResponse:
    """
    Fetch the content of a webpage using Serper.dev API.

    Args:
        url: The URL of the webpage to fetch
        include_markdown: Whether to include markdown formatting in the response (default: True)
        api_key: Serper API key (if not provided, will use SERPER_API_KEY env var)
        max_retries: Maximum number of retry attempts for retryable errors (default: 3)
        timeout: Request timeout in seconds (default: TIMEOUT)

    Returns:
        WebpageContentResponse containing:
        - text: The webpage content as plain text
        - markdown: The webpage content formatted as markdown (if include_markdown=True)
        - metadata: Additional metadata about the webpage
    """
    if not api_key:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError(
                "SERPER_API_KEY environment variable is not set or api_key parameter not provided"
            )

    scrape_url = "https://scrape.serper.dev"
    payload = json.dumps({"url": url, "includeMarkdown": include_markdown})
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    # Retry loop
    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(scrape_url, headers=headers, data=payload, timeout=timeout)

            # Handle non-200 status codes
            if response.status_code != 200:
                # Retry on server errors (500+) and gateway errors (502, 503, 504)
                retryable_status_codes = {502, 503, 504}
                is_server_error = response.status_code >= 500
                is_retryable = is_server_error or response.status_code in retryable_status_codes
                
                if is_retryable and attempt < max_retries - 1:
                    last_error = f"API error {response.status_code}: {response.text[:200]}"
                    wait_time = random.uniform(10, 15)
                    logger.warning("Serper scrape API error %d. Waiting %.1f seconds before retry %d/%d...", 
                                response.status_code, wait_time, attempt + 1, max_retries)
                    time.sleep(wait_time)
                    continue
                raise Exception(
                    f"API request failed with status {response.status_code}: {response.text[:500]}"
                )

            data = response.json()
            data["url"] = url
            return data

        except (ReadTimeout, Timeout, ConnectTimeout) as e:
            last_error = f"Request timed out after {timeout} seconds"
            if attempt < max_retries - 1:
                wait_time = random.uniform(10, 15)
                logger.warning("Serper scrape API timeout error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise Exception(f"{last_error}: {str(e)}")
        except ConnectionError as e:
            # Connection errors (network issues, upstream errors) should be retried
            last_error = f"Connection error: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = random.uniform(10, 15)
                logger.warning("Serper scrape API connection error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise Exception(f"Error fetching webpage content: {last_error}")
        except RequestException as e:
            # Other request exceptions should also be retried
            last_error = f"Request failed: {str(e)}"
            if attempt < max_retries - 1:
                wait_time = random.uniform(10, 15)
                logger.warning("Serper scrape API request error. Waiting %.1f seconds before retry %d/%d...", 
                            wait_time, attempt + 1, max_retries)
                time.sleep(wait_time)
                continue
            raise Exception(f"Error fetching webpage content: {last_error}")
        except json.JSONDecodeError as e:
            # JSON decode errors are not retryable
            raise Exception(f"Error parsing API response: {str(e)}")

    raise Exception(f"Serper scrape API request failed after {max_retries} attempts: {last_error or 'Unknown error'}")


# Example usage:
if __name__ == "__main__":
    # Regular search example
    try:
        results = search_serper("apple inc", num_results=5)
        print("Regular Search Results:")
        print(f"Found {len(results.get('organic', []))} results")
        if "knowledgeGraph" in results:
            print(f"Knowledge Graph: {results['knowledgeGraph']['title']}")
        print()
    except Exception as e:
        print(f"Search error: {e}")

    # Scholar search example
    try:
        scholar_results = search_serper_scholar(
            "attention is all you need", num_results=5
        )
        print("Scholar Search Results:")
        print(f"Found {len(scholar_results.get('organic', []))} academic papers")
        for paper in scholar_results.get("organic", [])[:2]:
            print(
                f"- {paper['title']} ({paper['year']}) - Cited by: {paper['citedBy']}"
            )
        print()
    except Exception as e:
        print(f"Scholar search error: {e}")
