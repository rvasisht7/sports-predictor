"""Agent layer (token-efficient version).
One tool returns everything the agent needs for the week in a compact form,
and the agent answers with JSON that Python logs itself. That keeps a run to
~2 API calls instead of 20+.

Setup:  export ANTHROPIC_API_KEY=sk-...
Usage:  python agent.py              # analyze the upcoming week
        python agent.py --smart      # use the larger model (more tokens, better reasoning)
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from data import current_week
from main import build
from model import build_features, explain, walk_forward
from stats import load_epa, load_injuries_current

FAST_MODEL = "claude-haiku-5-5"     # cheapest; fine for this task
SMART_MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 1500

LOG = Path(__file__).parent / "predictions_log.jsonl"
elo, games = build()
_cs = int(games.season.max())
_tg, _qg = load_epa(_cs)
_rep, _injv = load_injuries_current(_cs)
games, model = walk_forward(build_features(games, _tg, _qg, _injv))
games["model_home_prob"] = games["model_home_prob"].fillna(games["elo_home_prob"])


def _key_injuries(team, season, week):
    """Out/Doubtful players who matter, as short strings."""
    if _rep.empty:
        return []
    d = _rep[(_rep.season == season) & (_rep.week == week) & (_rep.team == team)
             & _rep.report_status.isin(["Out", "Doubtful"])
             & ((_rep.impact >= 0.3) | (_rep.position == "QB"))]
    return [f"{r.full_name} {r.position} {r.report_status}" for r in d.itertuples()]


def _last_qb_and_form(team: str, before_gameday: str, n: int = 3):
    """Last starting QB and recent W/L string for a team before a date."""
    g = games[games.completed & (games.gameday < before_gameday)
              & ((games.home_team == team) | (games.away_team == team))].tail(n)
    form, qb = "", None
    for r in g.itertuples():
        home = r.home_team == team
        us, them = (r.home_score, r.away_score) if home else (r.away_score, r.home_score)
        form += "W" if us > them else "L" if us < them else "T"
        qb = r.home_qb_name if home else r.away_qb_name
    return qb, form


def get_week_context():
    """Everything for the upcoming week's unplayed games, in short keys to save tokens."""
    season, week = current_week(games)
    wk = games[(games.season == season) & (games.week == week) & ~games.completed]
    rows = []
    for r in wk.itertuples():
        row = {"id": r.game_id, "g": f"{r.away_team}@{r.home_team}",
               "model": round(r.model_home_prob, 2),
               "top_factors": [f"{f['factor']} {f['points']:+.0f}" for f in explain(model, r._asdict())[1:4]],
               "vegas": None if r.vegas_home_prob != r.vegas_home_prob else round(r.vegas_home_prob, 2),
               "rest": f"{int(r.away_rest)}v{int(r.home_rest)}"}
        notes = []
        for side, team, qb in (("away", r.away_team, r.away_qb_name), ("home", r.home_team, r.home_qb_name)):
            last_qb, form = _last_qb_and_form(team, r.gameday)
            row[f"{side}_form"] = form
            if qb and last_qb and qb != last_qb:
                notes.append(f"{team} QB change: {last_qb} -> {qb}")
            out = _key_injuries(team, r.season, r.week)
            if out:
                notes.append(f"{team} out: " + ", ".join(out))
        if notes:
            row["notes"] = notes
        rows.append(row)
    return {"season": season, "week": week, "games": rows}


TOOLS = [{"name": "get_week_context",
          "description": "All unplayed games this week. model/vegas = home win prob; top_factors = biggest drivers in points for home; rest = away v home days; form = last 3 results, oldest first; notes = QB changes and key players out.",
          "input_schema": {"type": "object", "properties": {}}}]

SYSTEM = """NFL prediction agent. Call get_week_context once. Start from model (home win prob);
it already accounts for Elo, efficiency, QBs, injuries and rest. Adjust at most +/-0.05, only for
something it may miss (e.g. several key players out at one position). Usually keep it as is.
Reply with ONLY a JSON array, one object per game:
[{"id":"...","pick":"TEAM","home_prob":0.00,"why":"<=12 words"}]"""


def run(use_smart=False):
    client = anthropic.Anthropic()
    mdl = SMART_MODEL if use_smart else FAST_MODEL
    messages = [{"role": "user", "content": "Predict this week's games."}]
    total_in = total_out = 0
    while True:
        resp = client.messages.create(model=mdl, max_tokens=MAX_TOKENS, system=SYSTEM,
                                      tools=TOOLS, messages=messages)
        u = resp.usage
        total_in += u.input_tokens
        total_out += u.output_tokens
        print(f"  [tokens] this call: {u.input_tokens} in / {u.output_tokens} out "
              f"| running total: {total_in} in / {total_out} out")
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason != "tool_use":
            break
        results = [{"type": "tool_result", "tool_use_id": b.id,
                    "content": json.dumps(get_week_context(), separators=(",", ":"))}
                   for b in resp.content if b.type == "tool_use"]
        messages.append({"role": "user", "content": results})

    text = "".join(b.text for b in resp.content if b.type == "text")
    match = re.search(r"\[.*\]", text, re.S)
    if not match:
        print("Couldn't read the agent's answer:\n", text)
        return
    picks = json.loads(match.group())
    now = datetime.now().isoformat()
    with LOG.open("a") as f:
        for p in picks:
            f.write(json.dumps({"time": now, "model": mdl, **p}) + "\n")
    print()
    for p in picks:
        print(f"{p['id']:<20} {p['pick']:<4} home {p['home_prob']:.0%}  {p['why']}")
    print(f"\nLogged {len(picks)} picks to {LOG.name}.")


if __name__ == "__main__":
    run(use_smart="--smart" in sys.argv)
