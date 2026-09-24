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

OFFENSE = ["ppg", "o_ypp", "o_sr", "o_expl", "o_ppd", "o_rush_ypc", "o_pass_ypd",
           "o_third", "o_rz_td", "o_havoc_allowed"]
DEFENSE = ["opp_ppg", "d_ypp", "d_sr", "d_expl", "d_ppd", "d_rush_ypc", "d_pass_ypd",
           "d_third", "d_rz_td", "d_havoc"]
# offense stat -> the defensive stat it runs into
MATCH = [("o_ypp", "d_ypp"), ("o_sr", "d_sr"), ("o_expl", "d_expl"), ("o_ppd", "d_ppd"),
         ("o_rush_ypc", "d_rush_ypc"), ("o_pass_ypd", "d_pass_ypd"), ("o_third", "d_third"),
         ("o_rz_td", "d_rz_td"), ("o_havoc_allowed", "d_havoc")]
MATCH_LABEL = {"o_ypp": "Yards per play", "o_sr": "Success rate", "o_expl": "Explosive plays",
               "o_ppd": "Points per drive", "o_rush_ypc": "Rushing (yds/carry)",
               "o_pass_ypd": "Passing (yds/dropback)", "o_third": "3rd downs",
               "o_rz_td": "Red-zone TDs", "o_havoc_allowed": "Havoc (sacks, TFL, turnovers)"}


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
    return {"season": season, "meta": meta, "teams": rd("teams.parquet"),
            "games": rd("games.parquet"), "team_games": rd("team_games.parquet"),
            "season_stats": rd("team_season.parquet")}


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
    """Offensive vs defensive success rate by game."""
    g = tg_team.sort_values("week")
    x = [f"Wk {w}<br>{'vs' if h == 'home' else '@'} {opp_names.get(o, '?')}"
         for w, h, o in zip(g["week"], g["home_away"], g["opp_id"])]
    osr = g["o_success"] / g["o_succ_n"].where(g["o_succ_n"] > 0)
    dsr = g["d_success"] / g["d_succ_n"].where(g["d_succ_n"] > 0)
    fig = go.Figure()
    for name, y, c in (("Offense success rate", osr, BLUE), ("Defense success rate allowed", dsr, ORANGE)):
        fig.add_scatter(x=x, y=y * 100, name=name, mode="lines+markers",
                        line=dict(color=c, width=2), marker=dict(size=9, color=c),
                        hovertemplate="%{x}<br>" + name + ": %{y:.1f}%<extra></extra>")
    fig.add_hline(y=44, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text="FBS avg ~44%", annotation_position="bottom right")
    fig.update_yaxes(ticksuffix="%", gridcolor=GRID, title=None, rangemode="tozero")
    fig.update_xaxes(title=None)
    fig = _layout(fig, 300)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=0))
    return fig
