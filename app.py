"""CFB Quant Engine — research findings and the 2026 paper ledger.

Deployed on Streamlit Community Cloud from this repo. The app reads only files
committed here (research/FINDINGS.md, paper_ledger.csv); it never calls an API,
so it needs no keys. The password lives in the Streamlit Cloud app settings
(Secrets), never in git.
"""
import hmac
import os

import numpy as np
import pandas as pd
import streamlit as st

from wf import espn   # keyless ESPN client; imports nothing that needs API keys

st.set_page_config(page_title="CFB Quant Engine", page_icon="🏈", layout="wide")

ROOT = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(ROOT, "paper_ledger.csv")
FINDINGS = os.path.join(ROOT, "research", "FINDINGS.md")
BREAKEVEN = 110 / 210


# ------------------------------------------------------------ access

def check_password():
    expected = st.secrets.get("password") if hasattr(st, "secrets") else None
    if not expected:
        # Fail closed: no configured password means nobody gets in.
        st.title("🔒 Not configured")
        st.error("No password is set. Add `password = \"...\"` under "
                 "App settings → Secrets in Streamlit Community Cloud.")
        return False

    def entered():
        ok = hmac.compare_digest(st.session_state.get("pw", ""), str(expected))
        st.session_state["authed"] = ok
        st.session_state.pop("pw", None)

    if st.session_state.get("authed"):
        return True
    st.title("🔒 Restricted access")
    st.text_input("Password", type="password", on_change=entered, key="pw")
    if st.session_state.get("authed") is False:
        st.error("Incorrect password")
    return False


try:
    _authed = check_password()
except Exception:  # st.secrets raises when no secrets file exists at all
    st.title("🔒 Not configured")
    st.error("No password is set. Add `password = \"...\"` under "
             "App settings → Secrets in Streamlit Community Cloud.")
    _authed = False
if not _authed:
    st.stop()


# ------------------------------------------------------------ data

@st.cache_data(ttl=300)
def load_ledger():
    if not os.path.exists(LEDGER):
        return pd.DataFrame()
    try:
        df = pd.read_csv(LEDGER)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    if "price_assumed" in df.columns:
        df["price_assumed"] = df["price_assumed"].astype(str).str.lower().eq("true")
    else:
        df["price_assumed"] = True  # pre-flag rows came from CFBD's placeholder
    for c in ("commence", "logged_at"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


def summarise(g):
    w = int((g["result"] == "WIN").sum())
    l = int((g["result"] == "LOSS").sum())
    p = int((g["result"] == "PUSH").sum())
    n = w + l
    rate = w / n if n else np.nan
    se = np.sqrt(BREAKEVEN * (1 - BREAKEVEN) / n) if n else np.nan
    clv = g["clv_points"].dropna() if "clv_points" in g else pd.Series(dtype=float)
    return {
        "logged": len(g),
        "graded": w + l + p,
        "record": f"{w}-{l}-{p}",
        "win %": f"{rate*100:.1f}%" if n else "—",
        "±1 SD": f"{se*100:.1f} pts" if n else "—",
        "units": f"{g['profit_units'].sum():+.2f}" if n else "—",
        "mean CLV": f"{clv.mean():+.2f} pts" if len(clv) else "—",
        "beat close": f"{(clv > 0).mean()*100:.0f}%" if len(clv) else "—",
    }


@st.cache_data(ttl=600, show_spinner="Loading this week's games from ESPN...")
def load_board():
    """(season, week, DataFrame) for the slate ESPN is currently showing."""
    yr, wk, rows = espn.current_week()
    return yr, wk, pd.DataFrame(rows)


def fmt_price(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    p = int(p)
    return f"+{p}" if p > 0 else str(p)


def fmt_line(x, signed=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if x == 0 and signed:
        return "PK"
    return (f"{x:+g}" if signed else f"{x:g}")


def board_table(b, ledger):
    """One display row per game."""
    picks = {}
    if not ledger.empty:
        for _, r in ledger[ledger["result"].isna()].iterrows():
            picks.setdefault(r["game"], []).append(
                f"{r['side'].title() if r['market']=='total' else r['side']} "
                f"{r['line_taken']:g} ({fmt_price(r['price_taken'])})")
    out = []
    for _, g in b.iterrows():
        ko = pd.to_datetime(g["commence"], utc=True).tz_convert("America/New_York")
        hr = f"#{int(g['home_rank'])} " if pd.notna(g.get("home_rank")) and g.get("home_rank", 99) <= 25 else ""
        ar = f"#{int(g['away_rank'])} " if pd.notna(g.get("away_rank")) and g.get("away_rank", 99) <= 25 else ""
        sp, so = g.get("spread"), g.get("spread_open")
        tot, to = g.get("total"), g.get("total_open")
        moved_sp = "" if pd.isna(so) or pd.isna(sp) or so == sp else f"  (open {fmt_line(so)})"
        moved_t = "" if pd.isna(to) or pd.isna(tot) or to == tot else f"  (open {fmt_line(to, False)})"
        if not g.get("fbs_vs_fbs", True):
            flag = "vs FCS — out of scope"
        elif g["game"] in picks:
            flag = "📝 " + "; ".join(picks[g["game"]])
        elif pd.notna(tot) and tot >= 58:
            flag = "high total — under priced worse than −112"
        else:
            flag = ""
        score = ""
        if g.get("state") in ("in", "post") and pd.notna(g.get("home_score")):
            score = f"{int(g['away_score'])}–{int(g['home_score'])} {g.get('status','')}"
        out.append({
            "Kickoff (ET)": ko.strftime("%a %-I:%M %p"),
            "Matchup": f"{ar}{g['away']} @ {hr}{g['home']}" + (" (N)" if g.get("neutral") else ""),
            "Home spread": f"{fmt_line(sp)} ({fmt_price(g.get('home_spread_price'))}){moved_sp}",
            "Total": f"{fmt_line(tot, False)}  o{fmt_price(g.get('over_price'))} / u{fmt_price(g.get('under_price'))}{moved_t}",
            "Moneyline (A / H)": f"{fmt_price(g.get('away_ml'))} / {fmt_price(g.get('home_ml'))}",
            "Paper pick / note": flag,
            "Score": score,
            "TV": g.get("tv") or "",
            "_ko": ko,
        })
    return pd.DataFrame(out).sort_values("_ko").drop(columns="_ko")


# ------------------------------------------------------------ page

st.title("🏈 CFB Quant Engine")
st.warning(
    "**No proven edge.** The 2026 season is paper-traded only — **$0 at risk.** "
    "The 58.7% ATS this dashboard showed in 2025 was produced by a data leak; "
    "the honest figure was 50.7%."
)

df = load_ledger()
tab_week, tab_ledger, tab_findings, tab_about = st.tabs(
    ["This week", "Paper ledger", "Findings", "How to read this"])

with tab_week:
    try:
        yr, wk, board = load_board()
    except Exception as e:
        board = pd.DataFrame()
        st.error(f"Couldn't reach ESPN right now ({e.__class__.__name__}). Try again shortly.")
    if not board.empty:
        n_fbs = int(board["fbs_vs_fbs"].sum()) if "fbs_vs_fbs" in board else len(board)
        wk_led = df[(df["season"] == yr) & (df["week"] == wk)] if not df.empty else df
        a, b_, c = st.columns(3)
        a.metric(f"{yr} · Week {wk}", f"{len(board)} games")
        b_.metric("FBS vs FBS", n_fbs)
        c.metric("Paper picks this week", len(wk_led))
        only = st.toggle("Only show games with a paper pick", value=False)
        tbl = board_table(board, df)
        if only:
            tbl = tbl[tbl["Paper pick / note"].str.startswith("📝")]
        st.dataframe(tbl, hide_index=True, width="stretch", height=min(38 * (len(tbl) + 1), 900))
        st.caption(
            "Lines: DraftKings via ESPN, refreshed every 10 minutes. Spread is from the home "
            "team's side; prices in parentheses. 📝 marks paper bets in the ledger (no money). "
            "Candidate rule shown: UNDER when the total is 58+ and the under is priced −112 or "
            "better, FBS vs FBS only. It is being tracked, not proven — see Findings."
        )

with tab_ledger:
    if df.empty:
        st.info("No paper bets logged yet.")
    else:
        assumed = df["price_assumed"].sum()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bets logged", len(df))
        c2.metric("Graded", int(df["result"].notna().sum()))
        c3.metric("Closing line captured", int(df["close_line"].notna().sum()))
        c4.metric("Assumed prices", f"{int(assumed)} / {len(df)}")
        if assumed:
            st.caption(
                f"⚠️ {int(assumed)} bets were priced from a data source with no odds, so they "
                "carry an assumed −110. Their ROI is not a real number until a live odds "
                "feed is connected."
            )

        st.subheader("By strategy")
        rows = []
        for strat, g in df.groupby("strategy"):
            rows.append({"strategy": strat, **summarise(g)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(
            "Break-even at −110 is 52.38%. With under ~150 bets, ±1 SD is 4+ points, so "
            "win rate is mostly noise. Closing line value (CLV) is the early signal."
        )

        if "middle_id" in df.columns and df["middle_id"].notna().any():
            st.subheader("Total middles")
            m = df[df["middle_id"].notna()]
            pairs = [g for _, g in m.groupby("middle_id") if len(g) == 2]
            done = [g for g in pairs if g["result"].notna().all()]
            hit = sum(1 for g in done if (g["result"] == "WIN").all())
            a, b, c = st.columns(3)
            a.metric("Middles logged", len(pairs))
            b.metric("Hit both legs", f"{hit} / {len(done)}" if done else "—")
            c.metric("P&L", f"{sum(g['profit_units'].sum() for g in done):+.2f}u" if done else "—")
            st.caption("Historical 2019–25: 9.2% hit rate, +8.6% per middle.")

        st.subheader("All bets")
        show = df.sort_values("commence", ascending=False)
        cols = [c for c in ["commence", "week", "game", "strategy", "market", "side",
                            "line_taken", "price_taken", "book", "price_assumed",
                            "close_line", "clv_points", "result", "profit_units", "note"]
                if c in show.columns]
        st.dataframe(show[cols], hide_index=True, width="stretch")

        if df["logged_at"].notna().any():
            st.caption(f"Last logged: {df['logged_at'].max():%Y-%m-%d %H:%M} UTC")

with tab_findings:
    if os.path.exists(FINDINGS):
        st.markdown(open(FINDINGS).read())
    else:
        st.info("research/FINDINGS.md not found.")

with tab_about:
    st.markdown(
        """
**What this is.** A research project testing whether college football betting
markets can be beaten with public data. After two rounds of exhaustive testing
the answer so far is *no* — the market prices team quality better than any model
built here. Three narrow candidates are tracked in a paper ledger to gather
evidence without risking money.

**The weekly loop** (run locally, then commit the ledger):

```
python -m wf.paper log   2026 <week>   # Tue: record the best available number
python -m wf.paper close 2026 <week>   # Sat, before kickoff: capture the closing line
python -m wf.paper grade 2026          # Sun: settle results
```

**Reading the ledger.**
- *CLV (closing line value)* — how many points better than the final line you
  got. Consistently positive CLV is the precondition for long-run profit and
  shows up in weeks, not seasons.
- *Win %* — needs hundreds of bets to mean anything. Treat it as noise this year.
- *Assumed prices* — the free data source has no odds, so those bets use a
  placeholder −110 and their ROI is not real.
"""
    )
