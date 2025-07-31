from __future__ import annotations

"""
Open Notify – OO wrapper around the ISS tracking API.

Usage (CLI):
    python open_notify_client.py

Programmatic:
    client = OpenNotifyClient()
    ts, lat, lon, t = client.iss_now()
    crew, t2 = client.people_in_space()
"""

from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import requests


class OpenNotifyClient:
    """Lightweight client for https://open-notify.org API.

    Public helpers return `(payload, elapsed_seconds)` tuples so callers can
    expose timing metrics without extra plumbing.
    """

    BASE_URL: str = "http://api.open-notify.org"

    def __init__(self, timeout: float = 10.0, session: Optional[requests.Session] = None) -> None:
        """
        Initializes the instance with a specified timeout and a session. If no session is provided,
        a new requests.Session instance will be created.

        :param timeout: Timeout value in seconds for requests. Defaults to 10.0.
        :type timeout: float

        :param session: An optional `requests.Session` instance to manage HTTP requests. Defaults to None.
        :type session: Optional[requests.Session]
        """
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------
    # Low‑level HTTP helper
    # ------------------------------------------------------------------
    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], float]:
        """
        Performs an HTTP GET request to a given endpoint, measures the time taken for
        the request, and parses the JSON response.

        :param path: The relative URL path to which the GET request is made.
        :type path: str

        :param params: Optional query parameters to include in the GET request.
        :type params: Optional[Dict[str, Any]]

        :return: A tuple containing the parsed JSON response as a dictionary and the
                 elapsed time of the GET request in seconds.
        :rtype: Tuple[Dict[str, Any], float]
        """
        url = f"{self.BASE_URL}/{path.lstrip('/')}"

        t0 = perf_counter()
        resp = self.session.get(url, params=params, timeout=self.timeout)
        elapsed = perf_counter() - t0

        resp.raise_for_status()
        return resp.json(), elapsed

    # ------------------------------------------------------------------
    # High‑level API wrappers
    # ------------------------------------------------------------------
    def iss_now(self) -> Tuple[datetime, float, float, float]:
        """
        Retrieve the current location and time of the International Space Station (ISS).

        This method communicates with a specific API to fetch data on the current location
        of the ISS, including its latitude, longitude, and the timestamp of the related
        observation, as well as the time elapsed for the API call.

        :param self: Reference to the current instance of the class.
        :type self: OpenNotifyClient

        :return: A tuple consisting of:
            - **ts** (*datetime.datetime*): The UTC timestamp when the ISS location was recorded.
            - **lat** (*float*): The latitude of the ISS in degrees.
            - **lon** (*float*): The longitude of the ISS in degrees.
            - **elapsed** (*float*): The time in seconds it took to retrieve the data.
        """
        data, elapsed = self._get("iss-now.json")

        ts = datetime.fromtimestamp(int(data["timestamp"]), tz=timezone.utc)
        lat = float(data["iss_position"]["latitude"])
        lon = float(data["iss_position"]["longitude"])
        return ts, lat, lon, elapsed

    def people_in_space(self) -> Tuple[List[Dict[str, Any]], float]:
        """
        Retrieves the current list of people in space along with the time taken to fetch the data.

        The method fetches details of astronauts currently in space by making an API call to
        retrieve JSON data. The data contains information about individuals in space, including
        their names and the spacecraft they are on. Additionally, it tracks the elapsed time
        taken to perform the API call.

        :param self: An instance of the class where the method is defined.
        :type self: OpenNotifyClient

        :return: A tuple containing a list of dictionaries with details about people in space
                 and a float indicating the time elapsed during data retrieval.
        :rtype: Tuple[List[Dict[str, Any]], float]
        """
        data, elapsed = self._get("astros.json")
        return data["people"], elapsed


# ----------------------------------------------------------------------
# Command‑line entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Command-line entry point for the Open Notify client.
    This script initializes an instance of the OpenNotifyClient and retrieves
    the current location of the International Space Station (ISS) and the list of people in space
    along with the time taken for each request. It then prints this information to the console.
    """

    # Initialize the OpenNotifyClient
    client = OpenNotifyClient()

    # Retrieve the current ISS location and people in space
    ts, lat, lon, t_iss = client.iss_now()
    print(
        f"{ts.isoformat()} – ISS @ {lat:.2f}°, {lon:.2f}° "
        f"(query took {t_iss * 1000:.1f} ms)"
    )

    # Retrieve the list of people currently in space
    crew, t_people = client.people_in_space()
    print(f"\nPersonnel currently in space (queried in {t_people * 1000:.1f} ms):")
    for person in crew:
        print(f" • {person['name']} aboard {person['craft']}")
