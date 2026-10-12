"""Live layer: pulls scores from ESPN's public scoreboard feed and turns the
pregame Elo probability into a live, in-game win probability."""
import math
from statistics import NormalDist

import httpx

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}  # ESPN uses a few different abbreviations
SD = 13.5          # std. dev. of final NFL margin vs. expectation, in points
N = NormalDist()


def prob_to_spread(p: float) -> float:
    """Pregame win probability -> expected home margin in points."""
    return SD * N.inv_cdf(min(max(p, 0.001), 0.999))


def minutes_left(period: int, clock_seconds: float) -> float:
    if period <= 4:
        return (4 - period) * 15 + clock_seconds / 60
    return clock_seconds / 60  # overtime


def live_win_prob(pregame_p: float, margin: float, mins_left: float) -> float:
    """In-game win probability (Stern's model).
    The remaining game is treated as a slice of a normal distribution: the
    pregame expected margin shrinks with the time left, and so does the
    uncertainty (it scales with the square root of the time left)."""
    f = max(mins_left, 0) / 60
    if f < 0.0005:
        return 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    expected_rest = prob_to_spread(pregame_p) * f
    return N.cdf((margin + expected_rest) / (SD * math.sqrt(f)))


def _team(c):
    abbr = c["team"].get("abbreviation", "")
    return {
        "abbr": ESPN_TO_NFLVERSE.get(abbr, abbr),
        "name": c["team"].get("shortDisplayName") or c["team"].get("displayName", abbr),
        "score": int(c.get("score") or 0),
        "record": (c.get("records") or [{}])[0].get("summary", ""),
        "id": c["team"].get("id"),
    }


def parse_scoreboard(js: dict) -> list[dict]:
    games = []
    for ev in js.get("events", []):
        comp = ev["competitions"][0]
        sides = {c["homeAway"]: c for c in comp["competitors"]}
        status = comp.get("status") or ev.get("status", {})
        sit = comp.get("situation") or {}
        home, away = _team(sides["home"]), _team(sides["away"])
        poss = sit.get("possession")
        games.append({
            "id": ev["id"],
            "start": ev.get("date"),
            "state": status["type"]["state"],          # pre / in / post
            "detail": status["type"].get("shortDetail", ""),
            "period": status.get("period", 0),
            "clock": float(status.get("clock", 0) or 0),
            "neutral": bool(comp.get("neutralSite")),
            "home": home,
            "away": away,
            "possession": home["abbr"] if poss == home["id"] else away["abbr"] if poss == away["id"] else None,
            "down": sit.get("shortDownDistanceText") or sit.get("downDistanceText"),
            "red_zone": bool(sit.get("isRedZone")),
        })
    return games


async def fetch_scoreboard(client: httpx.AsyncClient) -> list[dict]:
    r = await client.get(ESPN_URL, timeout=10)
    r.raise_for_status()
    return parse_scoreboard(r.json())
