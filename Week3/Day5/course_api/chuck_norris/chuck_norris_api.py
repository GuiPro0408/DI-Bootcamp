from __future__ import annotations

"""
Chuck Norris – OO wrapper around the public jokes API.

Usage (CLI):
    python chuck_norris_api.py

Programmatic:
    client = ChuckNorrisClient()
    joke, jid, t = client.random_joke()
    cats, t2 = client.categories()
    hits, t3 = client.search("python")
"""

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import requests


def print_response_data(data: Any) -> None:
    """
    Print response data in a human-readable format with pretty formatting.

    :param data: The response data to print (can be dict, list, or str)
    """
    print("=" * 60)
    print("📄 API RESPONSE DATA")
    print("=" * 60)

    if isinstance(data, dict):
        _print_dict(data)

    elif isinstance(data, list):
        print(f"📋 List with {len(data)} item(s):")
        print("-" * 40)
        for i, item in enumerate(data, 1):
            print(f"\n🔹 Item {i}:")
            if isinstance(item, dict):
                _print_dict(item, indent="  ")
            else:
                print(f"  └─ {item}")

    elif isinstance(data, str):
        print(f"📝 String Response:")
        print(f"└─ {data}")

    else:
        print(f"🔍 Raw Data ({type(data).__name__}):")
        print(f"└─ {data}")

    print("=" * 60)


def _print_dict(data: dict, indent: str = "") -> None:
    """
    Helper function to print dictionary data with pretty formatting.

    :param data: Dictionary to print
    :param indent: Indentation string for nested items
    """
    for i, (key, value) in enumerate(data.items()):
        is_last = i == len(data) - 1
        connector = "└─" if is_last else "├─"

        # Format the value based on its type
        if isinstance(value, str):
            # Truncate very long strings for readability
            if len(value) > 100:
                formatted_value = f"{value[:97]}..."
            else:
                formatted_value = value
            print(f"{indent}{connector} {key}: \"{formatted_value}\"")
        elif isinstance(value, (int, float, bool)):
            print(f"{indent}{connector} {key}: {value}")
        elif isinstance(value, list):
            print(f"{indent}{connector} {key}: [List with {len(value)} item(s)]")
            if len(value) <= 3:  # Show small lists inline
                for j, item in enumerate(value):
                    sub_connector = "└─" if j == len(value) - 1 else "├─"
                    print(f"{indent}  {sub_connector} {item}")
        elif isinstance(value, dict):
            print(f"{indent}{connector} {key}: [Dictionary with {len(value)} key(s)]")
            # Don't recurse too deep to avoid clutter
        else:
            print(f"{indent}{connector} {key}: {value}")


class ChuckNorrisClient:
    """Lightweight client for https://api.chucknorris.io JSON endpoints."""

    BASE_URL: str = "https://api.chucknorris.io"

    def __init__(self, timeout: float = 10.0, session: Optional[requests.Session] = None) -> None:
        """
        :param timeout: per-request timeout in seconds (default 10 s)
        :param session: optional caller-supplied requests.Session for connection pooling
        """
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # Low-level HTTP helper
    # ------------------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], float]:
        """
        Sends a GET request to the specified path within the BASE_URL, including any
        optional query parameters. Returns the JSON-decoded response and the elapsed
        time of the request.

        :param path: The API endpoint path to send the GET request to. This is appended to the BASE_URL.
        :type path: str

        :param params: Optional query parameters to include in the GET request.
        :type params: Optional[Dict[str, Any]]

        :return: A tuple containing the JSON-decoded response as a dictionary and the elapsed time of the request in seconds.
        """
        url = f"{self.BASE_URL}/{path.lstrip('/')}"

        t0 = perf_counter()  # Start timing the request

        # Perform the GET request with the specified parameters and timeout
        resp = self.session.get(url, params=params, timeout=self.timeout)

        # Print data in a human-readable format
        if resp.status_code == 200:
            print(f"Request to {url} succeeded.")

            # Response data is printed in a pretty format
            try:
                data = resp.json()
                print_response_data(data)
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
        else:
            print(f"Request to {url} failed with status code {resp.status_code}.")

        elapsed = perf_counter() - t0

        resp.raise_for_status()
        return resp.json(), elapsed

    # ------------------------------------------------------------------
    # High-level API wrappers
    # ------------------------------------------------------------------
    def random_joke(self, category: Optional[str] = None) -> Tuple[str, str, float]:
        """
        Retrieve a random joke, optionally restricted to a category.

        :param category: e.g. 'dev', 'movie', None for truly random
        :return: (joke_text, joke_id, elapsed_seconds)
        """
        params = {"category": category} if category else None
        data, elapsed = self._get("jokes/random", params)
        return data["value"], data["id"], elapsed

    def categories(self) -> tuple[dict[str, Any], float]:
        """
        List all available joke categories.

        :return: (["animal", "dev", ...], elapsed_seconds)
        """
        data, elapsed = self._get("jokes/categories")
        return data, elapsed

    def search(self, query: str, limit: int = 20) -> Tuple[List[Dict[str, Any]], float]:
        """
        Free-text search for jokes.

        :param query: search term(s)
        :param limit: max jokes to return (API returns up to 20 by default)
        :return: ([{"id": ..., "value": ...}, ...], elapsed_seconds)
        """
        data, elapsed = self._get("jokes/search", params={"query": query})
        return data["result"][:limit], elapsed

    def joke_by_id(self, joke_id: str) -> Tuple[str, float]:
        """
        Fetch a specific joke by its UUID.

        :param joke_id: e.g. 'fTCHeXxcRGazFW9KRsRbJg'
        :return: (joke_text, elapsed_seconds)
        """
        data, elapsed = self._get(f"jokes/{joke_id}")
        return data["value"], elapsed


# ----------------------------------------------------------------------
# Command-line entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    client = ChuckNorrisClient()

    # Random joke
    joke, jid, t1 = client.random_joke()
    print(f"{datetime.now(timezone.utc).isoformat()} – {jid} ({t1 * 1000:.1f} ms)\n\"{joke}\"\n")

    # Categories
    cats, t2 = client.categories()
    print(f"Categories ({t2 * 1000:.1f} ms): {', '.join(cats)}\n")

    # Search example
    term = "python"
    hits, t3 = client.search(term)
    print(f"Top matches for '{term}' ({t3 * 1000:.1f} ms):")
    for h in hits:
        print(f" • {h['value']}")
