import json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUT = Path("model_output")
OUT.mkdir(exist_ok=True)

SCHED_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
TEAM_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_team/stats_team_week_{season}.csv"

TRAIN_START = 2006
TRAIN_END = 2022
VALID_START = 2023
VALID_END = 2025

def load_data():
    sched = pd.read_csv(SCHED_URL, low_memory=False)
    sched = sched[(sched.season.between(TRAIN_START, VALID_END)) & (sched.game_type == "REG")].copy()
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    sched = sched.dropna(subset=["result"]).copy()

    frames = []
    for season in range(TRAIN_START, VALID_END + 1):
        url = TEAM_URL.format(season=season)
        try:
            x = pd.read_csv(url, low_memory=False)
            frames.append(x)
            print("loaded", season, len(x))
        except Exception as e:
            print("team stats unavailable", season, e, file=sys.stderr)
    if not frames:
        raise RuntimeError("No team-stat files loaded.")
    team = pd.concat(frames, ignore_index=True)
    team = team[team["season_type"].eq("REG")].copy()
    return sched, team

def add_same_game_defense(team):
    # Offensive EPA per play for each team-game.
    team["off_plays"] = (
        pd.to_numeric(team.get("attempts"), errors="coerce").fillna(0)
        + pd.to_numeric(team.get("sacks_suffered"), errors="coerce").fillna(0)
        + pd.to_numeric(team.get("carries"), errors="coerce").fillna(0)
    ).replace(0, np.nan)
    team["off_epa"] = (
        pd.to_numeric(team.get("passing_epa"), errors="coerce").fillna(0)
        + pd.to_numeric(team.get("rushing_epa"), errors="coerce").fillna(0)
    )
    team["off_epa_play"] = team["off_epa"] / team["off_plays"]
    team["pass_epa_play"] = pd.to_numeric(team.get("passing_epa"), errors="coerce") / (
        pd.to_numeric(team.get("attempts"), errors="coerce")
        + pd.to_numeric(team.get("sacks_suffered"), errors="coerce")
    ).replace(0, np.nan)
    team["rush_epa_play"] = pd.to_numeric(team.get("rushing_epa"), errors="coerce") / pd.to_numeric(
        team.get("carries"), errors="coerce"
    ).replace(0, np.nan)
    ints = pd.to_numeric(team.get("passing_interceptions"), errors="coerce").fillna(0)
    rf = pd.to_numeric(team.get("rushing_fumbles_lost"), errors="coerce").fillna(0)
    sf = pd.to_numeric(team.get("sack_fumbles_lost"), errors="coerce").fillna(0)
    team["turnover_rate"] = (ints + rf + sf) / team["off_plays"]

    opp = team[["game_id","team","off_epa_play","pass_epa_play","rush_epa_play","turnover_rate"]].rename(
        columns={
            "team":"opponent_from_stats",
            "off_epa_play":"def_epa_allowed",
            "pass_epa_play":"def_pass_epa_allowed",
            "rush_epa_play":"def_rush_epa_allowed",
            "turnover_rate":"opp_turnover_rate",
        }
    )
    team = team.merge(opp, on="game_id", how="left")
    team = team[team["team"] != team["opponent_from_stats"]].copy()
    return team

def rolling_features(team, sched):
    dates = sched[["game_id","gameday"]]
    team = team.merge(dates, on="game_id", how="inner")
    team = team.sort_values(["team","gameday","game_id"]).copy()
    cols = [
        "off_epa_play","pass_epa_play","rush_epa_play","turnover_rate",
        "def_epa_allowed","def_pass_epa_allowed","def_rush_epa_allowed","opp_turnover_rate"
    ]
    for c in cols:
        s = pd.to_numeric(team[c], errors="coerce")
        # strictly pregame: shift before rolling
        team[c+"_r4"] = team.assign(_x=s).groupby("team")["_x"].transform(
            lambda z: z.shift(1).rolling(4, min_periods=2).mean()
        )
        team[c+"_r8"] = team.assign(_x=s).groupby("team")["_x"].transform(
            lambda z: z.shift(1).rolling(8, min_periods=4).mean()
        )
    keep = ["game_id","team"] + [c for c in team.columns if c.endswith("_r4") or c.endswith("_r8")]
    return team[keep]

def add_elo(sched):
    ratings = {}
    rows = []
    K = 20.0
    HFA = 55.0  # Elo points; modest HFA, zero on neutral field
    for _, g in sched.sort_values(["gameday","game_id"]).iterrows():
        h, a = g.home_team, g.away_team
        rh, ra = ratings.get(h,1500.0), ratings.get(a,1500.0)
        neutral = str(g.location).lower() == "neutral"
        adj_h = rh + (0 if neutral else HFA)
        exp_h = 1/(1+10**((ra-adj_h)/400))
        rows.append((g.game_id, rh, ra, rh-ra))
        result = float(g.result)
        actual = 1.0 if result>0 else (0.5 if result==0 else 0.0)
        mov = max(1.0, abs(result))
        mult = math.log(mov+1.0) * (2.2 / ((abs(adj_h-ra)*0.001)+2.2))
        delta = K * mult * (actual-exp_h)
        ratings[h] = rh + delta
        ratings[a] = ra - delta
    e = pd.DataFrame(rows, columns=["game_id","home_elo","away_elo","elo_diff"])
    return sched.merge(e, on="game_id", how="left")

def make_game_frame(sched, roll):
    s = add_elo(sched.copy())
    home = roll.rename(columns={c:"home_"+c for c in roll.columns if c not in ["game_id","team"]}).rename(columns={"team":"home_team_roll"})
    away = roll.rename(columns={c:"away_"+c for c in roll.columns if c not in ["game_id","team"]}).rename(columns={"team":"away_team_roll"})
    s = s.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")
    s = s[(s.home_team == s.home_team_roll) & (s.away_team == s.away_team_roll)].copy()

    base = [c[5:] for c in home.columns if c.startswith("home_") and c != "home_team_roll"]
    for c in base:
        if "home_"+c in s and "away_"+c in s:
            # For offense, home-away. For defensive EPA allowed, lower is better, so away-home.
            if c.startswith("def_"):
                s["diff_"+c] = s["away_"+c] - s["home_"+c]
            else:
                s["diff_"+c] = s["home_"+c] - s["away_"+c]

    s["rest_diff"] = pd.to_numeric(s.home_rest, errors="coerce") - pd.to_numeric(s.away_rest, errors="coerce")
    s["neutral"] = s.location.astype(str).str.lower().eq("neutral").astype(int)
    return s

def fit_and_eval(df):
    feature_cols = [c for c in df.columns if c.startswith("diff_")] + ["elo_diff","rest_diff","neutral"]
    train = df[df.season.between(TRAIN_START, TRAIN_END)].copy()
    valid = df[df.season.between(VALID_START, VALID_END)].copy()

    Xtr, ytr = train[feature_cols], train["result"].astype(float)
    Xv, yv = valid[feature_cols], valid["result"].astype(float)

    ridge = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", RidgeCV(alphas=np.logspace(-3,3,25)))
    ])
    gbr = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingRegressor(
            learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=25,
            l2_regularization=1.0, random_state=42
        ))
    ])

    models = {"ridge":ridge, "gradient_boosting":gbr}
    pred = valid[["game_id","season","week","gameday","away_team","home_team","result","spread_line"]].copy()

    summary=[]
    for name, model in models.items():
        model.fit(Xtr,ytr)
        p = model.predict(Xv)
        pred[name+"_margin"] = p
        summary.append({
            "model":name,
            "n":len(yv),
            "mae_margin":float(mean_absolute_error(yv,p)),
            "rmse_margin":float(mean_squared_error(yv,p)**0.5)
        })

    # Market implied home margin is +spread_line by nflverse convention.
    market_mask = pred.spread_line.notna()
    market_mae = mean_absolute_error(pred.loc[market_mask,"result"], pred.loc[market_mask,"spread_line"])
    market_rmse = mean_squared_error(pred.loc[market_mask,"result"], pred.loc[market_mask,"spread_line"])**0.5
    summary.append({"model":"market_closing_spread","n":int(market_mask.sum()),
                    "mae_margin":float(market_mae),"rmse_margin":float(market_rmse)})

    # ATS analysis: model picks home if predicted margin > spread, away otherwise.
    ats_rows=[]
    for name in models:
        pm = pred[name+"_margin"]
        pred["edge_"+name] = pm - pred["spread_line"]
        pred["ats_result_home"] = pred["result"] - pred["spread_line"]
        for season in [2023,2024,2025,"ALL"]:
            q = pred if season=="ALL" else pred[pred.season==season]
            for thr in [0,1,2,2.5,3,4,5]:
                z = q[q["spread_line"].notna() & (q["edge_"+name].abs() >= thr)].copy()
                if len(z)==0: continue
                side = np.where(z["edge_"+name] > 0, 1, -1)  # home=+1
                signed = z["ats_result_home"].to_numpy() * side
                wins = int((signed>0).sum()); losses=int((signed<0).sum()); pushes=int((signed==0).sum())
                denom=wins+losses
                ats_rows.append({
                    "model":name,"season":season,"edge_threshold":thr,"n":len(z),
                    "wins":wins,"losses":losses,"pushes":pushes,
                    "ats_pct_ex_push":wins/denom if denom else np.nan
                })

    pd.DataFrame(summary).to_csv(OUT/"model_summary.csv",index=False)
    ats = pd.DataFrame(ats_rows)
    ats.to_csv(OUT/"ats_by_edge.csv",index=False)
    pred.to_csv(OUT/"validation_predictions_2023_2025.csv",index=False)

    # Human-readable markdown
    lines=["# NFL retrospective model â first pass","",
           f"Training seasons: {TRAIN_START}-{TRAIN_END}; held-out validation: {VALID_START}-{VALID_END}.",
           "Betting spread was excluded from model inputs and used only after predictions were generated.","",
           "## Margin prediction","",
           "| Model | N | MAE | RMSE |","|---|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['model']} | {r['n']} | {r['mae_margin']:.3f} | {r['rmse_margin']:.3f} |")
    lines += ["","## ATS by model-market disagreement (all validation seasons)","",
              "| Model | Edge â¥ | N | W-L-P | ATS % (pushes excluded) |","|---|---:|---:|---:|---:|"]
    for _,r in ats[ats.season.astype(str).eq("ALL")].iterrows():
        lines.append(f"| {r.model} | {r.edge_threshold:g} | {int(r.n)} | {int(r.wins)}-{int(r.losses)}-{int(r.pushes)} | {100*r.ats_pct_ex_push:.1f}% |")
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")

    print((OUT/"REPORT.md").read_text())

if __name__=="__main__":
    sched, team = load_data()
    team = add_same_game_defense(team)
    roll = rolling_features(team, sched)
    games = make_game_frame(sched, roll)
    fit_and_eval(games)
