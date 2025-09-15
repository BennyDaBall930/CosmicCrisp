import aiohttp
from aiohttp.client_exceptions import ClientError, ContentTypeError
from python.helpers import dotenv

# Allow override via .env `SEARXNG_URL`; default to local dev server on 8888
# Example: SEARXNG_URL=http://127.0.0.1:8888/search
URL = dotenv.get_dotenv_value("SEARXNG_URL", "http://127.0.0.1:8888/search")

async def search(query: str):
    return await _search(query=query)

async def _search(query: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(URL, params={"q": query, "format": "json"}) as response:
                # Fail fast on HTTP errors
                if response.status != 200:
                    text = await response.text()
                    return Exception(f"SearXNG HTTP {response.status}: {text[:200]}")

                # Ensure JSON content
                ctype = response.headers.get("Content-Type", "").lower()
                if "application/json" not in ctype:
                    text = await response.text()
                    return Exception(
                        f"SearXNG returned non-JSON ({ctype}). Sample: {text[:200]}"
                    )

                try:
                    return await response.json()
                except ContentTypeError as e:
                    text = await response.text()
                    return Exception(f"SearXNG JSON decode error: {e}; Sample: {text[:200]}")
    except ClientError as e:
        return Exception(f"SearXNG request error: {e}")
    except Exception as e:
        return e
