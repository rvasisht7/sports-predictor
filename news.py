"""News: automatic NFL headlines from ESPN, plus your own stories saved in
data/news.json (commit that file so your stories show up on the deployed site)."""
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from teams import TEAM_ABBR

ESPN_NEWS = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/news"
MY_STORIES = Path(__file__).parent / "data" / "news.json"
CACHE_SECONDS = 15 * 60
_cache = {"t": 0, "items": [], "error": None}


def parse_espn(js: dict) -> list[dict]:
    items = []
    for a in js.get("articles", []):
        teams = sorted({TEAM_ABBR[c["description"]] for c in a.get("categories", [])
                        if c.get("type") == "team" and c.get("description") in TEAM_ABBR})
        link = (a.get("links") or {}).get("web", {}).get("href")
        if not link or not a.get("headline"):
            continue
        img = next((i.get("url") for i in a.get("images", []) if i.get("url")), None)
        items.append({"title": a["headline"], "summary": a.get("description", ""), "url": link,
                      "source": "ESPN", "date": a.get("published"), "teams": teams, "image": img})
    return items


async def espn_headlines(client: httpx.AsyncClient) -> dict:
    if time.time() - _cache["t"] > CACHE_SECONDS:
        try:
            r = await client.get(ESPN_NEWS, params={"limit": 50}, timeout=10)
            r.raise_for_status()
            _cache.update(t=time.time(), items=parse_espn(r.json()), error=None)
        except Exception as e:
            print(f"news error: {e!r}")
            _cache.update(t=time.time(), error="Can't reach ESPN for headlines right now.")
    return {"items": _cache["items"], "error": _cache["error"]}


def my_stories() -> list[dict]:
    if not MY_STORIES.exists():
        return []
    return sorted(json.loads(MY_STORIES.read_text()), key=lambda s: s.get("date", ""), reverse=True)


def _save(stories):
    MY_STORIES.write_text(json.dumps(stories, indent=2))


def add_story(title, url, summary="", source="", teams=None) -> dict:
    s = {"id": uuid.uuid4().hex[:8], "title": title.strip(), "url": url.strip(),
         "summary": summary.strip(), "source": source.strip(),
         "teams": sorted({t.strip().upper() for t in (teams or []) if t.strip()}),
         "date": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    stories = my_stories()
    stories.append(s)
    _save(stories)
    return s


def delete_story(story_id: str) -> bool:
    stories = my_stories()
    keep = [s for s in stories if s["id"] != story_id]
    _save(keep)
    return len(keep) != len(stories)
