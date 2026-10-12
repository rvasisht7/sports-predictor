# Sports Predictor

NFL win probabilities with a live dashboard: live scores and in-game win chances,
weekly picks with a breakdown of why, team pages, season stat leaders, fantasy start/sit projections, a graded prediction tracker,
injury reports, standings, playoff odds with a week-by-week bracket, team tendencies,
favorite teams and notifications.

## Run locally
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    uvicorn app:app --reload
Open http://127.0.0.1:8000

## The model (model.py)
Logistic regression trained walk-forward (each season predicted only from earlier seasons) on:
Elo rating, team EPA per play, starting-QB EPA per dropback, QB changes, injuries
(Out/Doubtful starters weighted by position and snap share), rest, neutral site.
Backtest 2015-2026: Brier 0.218 (Elo alone 0.222, Vegas 0.213).

## Your news stories
Open the News tab on your own computer to add stories with the form (it's hidden on the
deployed site). They're saved in data/news.json; commit and push that file to publish them.

## Files
- app.py: web server + API, refreshes data every 6 h, live scores every 30 s
- model.py: features + trained model     stats.py: play-by-play EPA, QB stats, injuries
- elo.py: Elo ratings    live.py: ESPN scores + in-game win probability
- season.py: standings, season simulation, playoff bracket    tendencies.py: team tendencies + matchup keys
- teams.py: team names, conferences, divisions
- news.py: ESPN headlines + your saved stories
- players.py: season leaders and fantasy projections (tuned and tested on 2024-2025)
- data.py, main.py: schedules + command-line backtest     agent.py: Claude agent (uses API tokens)
- data/: condensed 2006-2025 stats (commit this). Rebuild with: python stats.py 2025
- static/: the website pages

## Deploy (Render)
Push to GitHub, then on render.com: New > Blueprint > pick the repo (uses render.yaml).
