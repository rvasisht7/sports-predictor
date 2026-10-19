"""Ask: answers questions like "which team is best on the road?" using Claude,
restricted to tools that read this site's own data, so answers use the same
numbers as every page.

Cost controls (the site is public): cheapest model, short answers, at most
4 tool rounds per question, a per-visitor hourly limit and a site-wide daily cap.
Set ANTHROPIC_API_KEY on the server to turn it on."""
import json
import os
import time
from collections import defaultdict, deque

MODEL = os.environ.get("ASK_MODEL", "claude-haiku-5-5")
PER_VISITOR_PER_HOUR = int(os.environ.get("ASK_PER_HOUR", 8))
DAILY_CAP = int(os.environ.get("ASK_DAILY_LIMIT", 200))
MAX_ROUNDS = 4
MAX_QUESTION = 300

_visits = defaultdict(deque)
_day = {"date": None, "count": 0}

SYSTEM = """You answer questions about the current NFL season for a student project website.
Use the tools to look up facts; never rely on memory for stats, records or results.
Answer in 2-5 short sentences or a short list. Lead with the direct answer, then the key numbers.
If the data doesn't cover the question, say so plainly. Stay on NFL topics.
Never give betting advice: if asked what to bet, say the site is for learning, not betting, and offer the model's view instead.
Use plain text with at most simple "- " bullets and **bold** for names."""

TOOLS = [
    {"name": "standings_and_splits",
     "description": "Every team's record, division, home and away records with average margin, points for/against per game, streak, and playoff/division chances. Use for 'best on the road', 'best home team', 'best defense by points allowed', etc.",
     "input_schema": {"type": "object", "properties": {}}},
    {"name": "team_profile",
     "description": "One team in depth: Elo rating and rank, record, efficiency ranks, tendencies, season outlook, next game with win chance, notable injuries.",
     "input_schema": {"type": "object", "properties": {"team": {"type": "string", "description": "Abbreviation (KC) or name (Chiefs)"}}, "required": ["team"]}},
    {"name": "stat_leaders",
     "description": "Season leaders for a stat, e.g. 'passing yards', 'sacks', 'receptions', 'rushing TDs', 'interceptions', 'field goal %', 'fantasy points'. Optional team filter.",
     "input_schema": {"type": "object", "properties": {"stat": {"type": "string"}, "team": {"type": "string"},
                                                        "n": {"type": "integer"}}, "required": ["stat"]}},
    {"name": "player_lookup",
     "description": "A player's season stats, league ranks, recent games and next fantasy projection with start/sit call.",
     "input_schema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "games",
     "description": "Games with the model's pick, win chance and result. Filter by week and/or team. Defaults to the current week.",
     "input_schema": {"type": "object", "properties": {"week": {"type": "integer"}, "team": {"type": "string"}}}},
    {"name": "playoff_odds",
     "description": "Each team's chances to make the playoffs, win its division, get a bye, reach and win the Super Bowl, from 10,000 simulations.",
     "input_schema": {"type": "object", "properties": {"conference": {"type": "string", "enum": ["AFC", "NFC"]}}}},
    {"name": "fantasy_rankings",
     "description": "This week's fantasy projections (PPR) with start/sit verdicts. Optional position filter (QB, RB, WR, TE).",
     "input_schema": {"type": "object", "properties": {"position": {"type": "string"}, "n": {"type": "integer"}}}},
    {"name": "matchup",
     "description": "How two teams match up: tendencies-based keys for each side, and the model's win chance if they play this season.",
     "input_schema": {"type": "object", "properties": {"team_a": {"type": "string"}, "team_b": {"type": "string"}},
                      "required": ["team_a", "team_b"]}},
]


def enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def check_limits(visitor: str):
    """Returns an error message if this visitor or the whole site is over its limit."""
    now = time.time()
    today = time.strftime("%Y-%m-%d")
    if _day["date"] != today:
        _day.update(date=today, count=0)
    if _day["count"] >= DAILY_CAP:
        return "Ask has hit its daily limit for the site. Try again tomorrow."
    q = _visits[visitor]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= PER_VISITOR_PER_HOUR:
        return f"You've asked {PER_VISITOR_PER_HOUR} questions this hour. Try again a little later."
    q.append(now)
    _day["count"] += 1
    return None


def answer(question: str, tools: dict, history: list | None = None) -> dict:
    """Runs the tool loop. `tools` maps tool names to Python functions."""
    import anthropic
    client = anthropic.Anthropic()
    messages = []
    for h in (history or [])[-4:]:               # a little context for follow-ups
        if h.get("q") and h.get("a"):
            messages += [{"role": "user", "content": h["q"][:MAX_QUESTION]},
                         {"role": "assistant", "content": h["a"][:1200]}]
    messages.append({"role": "user", "content": question[:MAX_QUESTION]})
    used = []
    for _ in range(MAX_ROUNDS + 1):
        resp = client.messages.create(model=MODEL, max_tokens=700, system=SYSTEM, tools=TOOLS, messages=messages)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            text = "".join(b.text for b in resp.content if b.type == "text").strip()
            return {"answer": text or "I couldn't find an answer to that.", "used": used}
        results = []
        for b in resp.content:
            if b.type != "tool_use":
                continue
            used.append(b.name)
            try:
                out = tools[b.name](**(b.input or {}))
            except Exception as e:
                out = {"error": f"{type(e).__name__}: {e}"}
            results.append({"type": "tool_result", "tool_use_id": b.id,
                            "content": json.dumps(out, default=str, separators=(",", ":"))[:12000]})
        messages.append({"role": "user", "content": results})
    return {"answer": "That question needed more lookups than allowed. Try asking something more specific.", "used": used}
