"""Shared UI helpers for the stats site (Streamlit)."""
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import source

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Validated two-slot palette (dataviz reference instance, light + dark pass).
BLUE, ORANGE = "#2a78d6", "#eb6834"      # away / above-average  |  home / below-average
GRID, MUTED = "rgba(128,128,128,0.18)", "#8a8984"

# Stat groups shown as percentile profiles on team pages
PROFILE = {
    "Offense": ["o_epa", "o_rush_epa", "o_pass_epa", "o_early_epa", "o_sr", "o_sd_sr", "o_pd_sr",
                "o_isoppp", "o_expl", "o_stuff", "o_third", "o_havoc_allowed"],
    "Defense": ["d_epa", "d_rush_epa", "d_pass_epa", "d_early_epa", "d_sr", "d_sd_sr", "d_pd_sr",
                "d_isoppp", "d_expl", "d_stuff", "d_third", "d_havoc"],
    "Drives & field position": ["o_ppd", "o_pts_opp", "o_rz_td", "o_start",
                                "d_ppd", "d_pts_opp", "d_rz_td", "d_start"],
    "Results": ["margin_pg", "ppg", "opp_ppg", "to_margin_pg", "penalty_yds_pg"],
}
OFFENSE, DEFENSE = PROFILE["Offense"], PROFILE["Defense"]

# offense stat -> the defensive stat it runs into (unit vs unit)
MATCH = [("o_epa", "d_epa"), ("o_rush_epa", "d_rush_epa"), ("o_pass_epa", "d_pass_epa"),
         ("o_early_epa", "d_early_epa"), ("o_sd_sr", "d_sd_sr"), ("o_pd_sr", "d_pd_sr"),
         ("o_isoppp", "d_isoppp"), ("o_pts_opp", "d_pts_opp"), ("o_stuff", "d_stuff"),
         ("o_havoc_allowed", "d_havoc"), ("o_third", "d_third")]
MATCH_LABEL = {"o_epa": "Overall efficiency (EPA/play)", "o_rush_epa": "Running game (EPA/rush)",
               "o_pass_epa": "Passing game (EPA/dropback)", "o_early_epa": "Early downs (EPA)",
               "o_sd_sr": "Standard downs (success)", "o_pd_sr": "Passing downs (success)",
               "o_isoppp": "Big plays (EPA per success)", "o_pts_opp": "Finishing drives (pts/opportunity)",
               "o_stuff": "Run blocking vs front (stuff rate)",
               "o_havoc_allowed": "Protection vs havoc", "o_third": "3rd downs"}

SIGMA_MARGIN = 15.4   # sd of actual margin around the line (CFB, 2018-25); for win prob


# ------------------------------------------------------------ data

@st.cache_data(ttl=900)
def snapshot(season=None):
    base = os.path.join(ROOT, "site_data")
    if not os.path.isdir(base):
        return None
    seasons = sorted(int(x) for x in os.listdir(base) if x.isdigit())
    if not seasons:
        return None
    season = season or seasons[-1]
    d = os.path.join(base, str(season))
    rd = lambda n: pd.read_parquet(os.path.join(d, n))
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    players = {k: rd(f"players_{k}.parquet") for k in ("passing", "rushing", "receiving")
               if os.path.exists(os.path.join(d, f"players_{k}.parquet"))}
    return {"season": season, "meta": meta, "teams": rd("teams.parquet"),
            "games": rd("games.parquet"), "team_games": rd("team_games.parquet"),
            "season_stats": rd("team_season.parquet"), "players": players}


@st.cache_data(ttl=600, show_spinner="Loading this week from ESPN...")
def live_week():
    from wf import espn
    yr, wk, rows = espn.current_week()
    return yr, wk, pd.DataFrame(rows)


@st.cache_data(ttl=600, show_spinner="Loading game preview from ESPN...")
def live_summary(event_id):
    return source.summary(event_id, completed=False)


# ------------------------------------------------------------ formatting

def fmt(v, kind):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if kind == "pct":
        return f"{v*100:.1f}%"
    if kind == "+pct":
        return f"{v*100:+.1f} pts"
    if kind.startswith("+"):
        return f"{v:+.{kind[1]}f}"
    return f"{v:.{kind[0]}f}"


def ordinal(n):
    n = int(n)
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1:'st',2:'nd',3:'rd'}.get(n % 10, 'th')}"


def price(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    p = int(p)
    return f"+{p}" if p > 0 else str(p)


def line(x, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if signed and x == 0:
        return "PK"
    return f"{x:+g}" if signed else f"{x:g}"


def stat_cell(row, col, meta):
    m = meta["stats"][col]
    v = fmt(row.get(col), m["fmt"])
    r = row.get(f"{col}_rank")
    return f"{v}  ({ordinal(r)})" if r is not None and not pd.isna(r) else v


def win_prob(margin):
    """P(team wins) given a projected margin, normal approx with CFB spread sd."""
    from math import erf, sqrt
    return 0.5 * (1 + erf(margin / (SIGMA_MARGIN * sqrt(2))))


def rank_label(r):
    try:
        r = int(r)
    except (TypeError, ValueError):
        return ""
    return f"#{r} " if 0 < r <= 25 else ""


# ------------------------------------------------------------ charts

def _layout(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), showlegend=False, bargap=0.35,
        hoverlabel=dict(font_size=13),
    )
    return fig


def percentile_bars(row, cols, meta, n_teams):
    """Horizontal percentile bars (100 = best in FBS). Blue = better than the
    median, orange = worse; value and national rank are direct-labelled."""
    labels, pct, text, hover, colors = [], [], [], [], []
    for c in cols:
        m = meta["stats"][c]
        p = row.get(f"{c}_pctl")
        if p is None or pd.isna(p):
            continue
        labels.append(m["label"])
        pct.append(p)
        colors.append(BLUE if p >= 50 else ORANGE)
        val = fmt(row.get(c), m["fmt"])
        rk = ordinal(row.get(f"{c}_rank"))
        text.append(f"{val} · {rk}")
        hover.append(f"<b>{m['label']}</b><br>{val}<br>{rk} of {n_teams} FBS teams"
                     f"<br>better than {p:.0f}% of FBS")
    fig = go.Figure(go.Bar(
        x=pct, y=labels, orientation="h", marker=dict(color=colors, line=dict(width=0)),
        text=text, textposition="outside", cliponaxis=False,
        hovertext=hover, hoverinfo="text"))
    fig.add_vline(x=50, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(range=[0, 125], showgrid=True, gridcolor=GRID, zeroline=False,
                     tickvals=[0, 25, 50, 75, 100], ticksuffix="", title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    return _layout(fig, 34 * len(labels) + 40)


def matchup_bars(off_row, def_row, off_name, def_name, off_color, def_color, meta):
    """Paired percentile bars: one team's offense vs the other's defense."""
    labels, o, d, oh, dh = [], [], [], [], []
    for oc, dc in MATCH:
        op, dp = off_row.get(f"{oc}_pctl"), def_row.get(f"{dc}_pctl")
        if op is None or dp is None or pd.isna(op) or pd.isna(dp):
            continue
        labels.append(MATCH_LABEL[oc])
        o.append(op); d.append(dp)
        oh.append(f"<b>{off_name} offense</b><br>{MATCH_LABEL[oc]}: "
                  f"{fmt(off_row.get(oc), meta['stats'][oc]['fmt'])} ({ordinal(off_row.get(oc+'_rank'))})")
        dh.append(f"<b>{def_name} defense</b><br>{MATCH_LABEL[oc]} allowed: "
                  f"{fmt(def_row.get(dc), meta['stats'][dc]['fmt'])} ({ordinal(def_row.get(dc+'_rank'))})")
    fig = go.Figure()
    fig.add_bar(name=f"{off_name} offense", y=labels, x=o, orientation="h",
                marker=dict(color=off_color, line=dict(width=0)),
                hovertext=oh, hoverinfo="text", text=[f"{v:.0f}" for v in o], textposition="outside",
                cliponaxis=False)
    fig.add_bar(name=f"{def_name} defense", y=labels, x=d, orientation="h",
                marker=dict(color=def_color, line=dict(width=0)),
                hovertext=dh, hoverinfo="text", text=[f"{v:.0f}" for v in d], textposition="outside",
                cliponaxis=False)
    fig.add_vline(x=50, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_xaxes(range=[0, 112], showgrid=True, gridcolor=GRID, zeroline=False,
                     tickvals=[0, 25, 50, 75, 100], title="FBS percentile (100 = best)")
    fig.update_yaxes(autorange="reversed", title=None)
    fig = _layout(fig, 52 * len(labels) + 70)
    fig.update_layout(showlegend=True, barmode="group", bargroupgap=0.08,
                      legend=dict(orientation="h", y=1.02, x=0, yanchor="bottom"))
    return fig


def game_log_chart(tg_team, opp_names):
    """Offensive EPA/play vs defensive EPA/play allowed, by game."""
    g = tg_team.sort_values("week")
    x = [f"Wk {w}<br>{'vs' if h == 'home' else '@'} {opp_names.get(o, 'FCS opp.')}"
         for w, h, o in zip(g["week"], g["home_away"], g["opp_id"])]
    oe = g["o_epa"] / g["o_epa_n"].where(g["o_epa_n"] > 0)
    de = g["d_epa"] / g["d_epa_n"].where(g["d_epa_n"] > 0)
    fig = go.Figure()
    for name, y, c in (("Offense EPA/play", oe, BLUE), ("Defense EPA/play allowed", de, ORANGE)):
        fig.add_scatter(x=x, y=y, name=name, mode="lines+markers",
                        line=dict(color=c, width=2), marker=dict(size=9, color=c),
                        hovertemplate="%{x}<br>" + name + ": %{y:+.3f}<extra></extra>")
    fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text="average", annotation_position="bottom right")
    fig.update_yaxes(gridcolor=GRID, title=None, tickformat="+.2f")
    fig.update_xaxes(title=None)
    fig = _layout(fig, 300)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0))
    return fig


def efficiency_map(ss, highlight=None, conf=None):
    """Every FBS team by adjusted offense (x) and adjusted defense (y, flipped so
    up = better). Top-right = good at both. One hue; the highlighted team and an
    optional conference are drawn in the accent colour, everyone else muted."""
    d = ss.dropna(subset=["adj_off", "adj_def"]).copy()
    focus = d["team_id"].eq(highlight) | (d["conference"].eq(conf) if conf else False)
    fig = go.Figure()
    for mask, color, size, show in ((~focus, "rgba(138,137,132,0.45)", 8, False), (focus, BLUE, 11, True)):
        dd = d[mask]
        fig.add_scatter(
            x=dd["adj_off"], y=dd["adj_def"], mode="markers+text" if show else "markers",
            text=dd["team"] if show else None, textposition="top center", textfont=dict(size=11),
            marker=dict(size=size, color=color, line=dict(width=2, color="rgba(255,255,255,0.9)")),
            customdata=np.stack([dd["team"], dd["record"], dd["power"], dd["power_rank"]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>Adj. offense %{x:+.3f}"
                          "<br>Adj. defense %{y:+.3f}<br>Power %{customdata[2]:+.1f} "
                          "(#%{customdata[3]})<extra></extra>")
    mo, md = d["adj_off"].median(), d["adj_def"].median()
    fig.add_vline(x=mo, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_hline(y=md, line=dict(color=MUTED, width=1, dash="dot"))
    for tx, ty, lab, xa, ya in ((1, 1, "Good offense, good defense", "right", "top"),
                                (0, 1, "Defense-first", "left", "top"),
                                (1, 0, "Offense-first", "right", "bottom"),
                                (0, 0, "Struggling on both sides", "left", "bottom")):
        fig.add_annotation(xref="paper", yref="paper", x=tx, y=ty, text=lab, showarrow=False,
                           font=dict(size=11, color=MUTED), xanchor=xa, yanchor=ya)
    fig.update_xaxes(title="Adjusted offense (EPA/play) →  better", gridcolor=GRID, zeroline=False,
                     tickformat="+.2f")
    fig.update_yaxes(title="Adjusted defense (EPA/play allowed) →  better", gridcolor=GRID,
                     zeroline=False, autorange="reversed", tickformat="+.2f")
    return _layout(fig, 560)


# ------------------------------------------------------------ players

PLAYER_VIEWS = {
    "passing": {"vol": "dropbacks", "label": "Quarterbacks", "min_pg": 12,
                "cols": [("dropbacks", "Dropbacks", "%d"), ("epa_per", "EPA/dropback", "%+.3f"),
                         ("epa", "Total EPA", "%+.1f"), ("success", "Success", "pct"),
                         ("comp_pct", "Comp %", "pct"), ("ypa", "Yds/att", "%.1f"),
                         ("td", "TD", "%d"), ("ints", "INT", "%d"), ("sacks", "Sacks", "%d")]},
    "rushing": {"vol": "carries", "label": "Rushers", "min_pg": 6,
                "cols": [("carries", "Carries", "%d"), ("epa_per", "EPA/carry", "%+.3f"),
                         ("epa", "Total EPA", "%+.1f"), ("success", "Success", "pct"),
                         ("ypc", "Yds/carry", "%.2f"), ("yds", "Yards", "%d"),
                         ("explosive", "12+ yd runs", "%d"), ("td", "TD", "%d")]},
    "receiving": {"vol": "targets", "label": "Receivers", "min_pg": 3,
                  "cols": [("targets", "Targets", "%d"), ("epa_per", "EPA/target", "%+.3f"),
                           ("epa", "Total EPA", "%+.1f"), ("success", "Success", "pct"),
                           ("rec", "Rec", "%d"), ("catch_pct", "Catch %", "pct"),
                           ("yds", "Yards", "%d"), ("ypt", "Yds/target", "%.1f"), ("td", "TD", "%d")]},
}


def player_table(df, kind, sort_by="epa", team=True):
    """Display frame + column_config for a player table."""
    v = PLAYER_VIEWS[kind]
    d = df.sort_values(sort_by, ascending=False).copy()
    out = pd.DataFrame({"Player": d["player"].values})
    if team:
        out.insert(0, "Logo", d["logo"].values)
        out["Team"] = d["team"].values
    cfg = {"Logo": st.column_config.ImageColumn("", width="small")}
    for c, lab, f in v["cols"]:
        if f == "pct":
            out[lab] = (d[c] * 100).round(1).values
            cfg[lab] = st.column_config.NumberColumn(lab, format="%.1f%%")
        else:
            out[lab] = d[c].values
            cfg[lab] = st.column_config.NumberColumn(lab, format=f)
    return out, cfg
