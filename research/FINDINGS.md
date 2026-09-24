# Where is the edge? — findings, Aug–Sep 2026

**Verdict: no proven edge.** Three candidates are worth tracking; none is proven.
The 2026 season is paper-traded only. **$0 at risk.**

## How we got here

The 2025 dashboard reported **58.7% against the spread**. That number was fake.
The model was fed each team's *full-season* SRS rating, which is computed from
the results of the very games being predicted. Rebuilt honestly (train on 2024,
test on 2025, n = 1,584), the same model hit **50.7%** — the same as betting
every home team. Break-even at −110 is **52.38%**.

The search that followed:

- A walk-forward harness where every feature is provably known before kickoff,
  plus an audit that fails if leakage creeps back in.
- 40 pre-registered situational hypotheses, corrected for multiple testing.
- Two rounds of parallel research (24 lanes, roughly 700 hypotheses) covering
  every CollegeFootballData source: play-by-play, drives, recruiting, coaching,
  QB continuity, special teams, opponent adjustment, TV exposure, line movement,
  quarter scores, teasers, cross-book pricing.

## What is priced (the market already knows it)

**Team quality, in every form.** Every rating built here — drive efficiency,
opponent-adjusted efficiency, garbage-time-excluded stats, special teams,
recruiting, recent form, play-by-play profiles — correlates **0.80–0.91 with the
betting line** and **≤ 0.043 with whether teams cover**. The sharpest case: a
properly tuned opponent-adjusted rating tracked the line at r = 0.906, and
betting it where it disagreed with the line got *worse* as the disagreement grew
(49.7% → 44.4%). The line is the better number exactly where they differ.

**Everything situational.** Big favorites, home underdogs, rest and byes,
travel, altitude, domes, weeknight games, look-ahead and let-down spots,
coaching changes, QB changes, turnover and close-game luck, transfer portal
churn, public attention (high-profile games are priced *more* accurately),
following or fading line movement.

**Teasers.** The classic 6-point teaser loses in college football: 70.8% per
leg against 72.4% needed (n = 1,344, 13 seasons).

**The model.** Forecasting the final margin, the best model misses by 12.5
points on average; the closing line misses by 12.19. The model is worse than
the line.

## The three live candidates

| # | Candidate | Evidence | Why it isn't proven |
|---|---|---|---|
| 1 | **Total middles.** When two books' totals differ by ≥ 2.5 points, bet the over at the lower one and the under at the higher one. If the score lands between, both win. | 2019–25: 261 middles (~37 a season), +8.6% expected per middle, 95% CI **+2.0% to +15.3%** — the only claim whose interval excludes zero. Needs no prediction. | Never went through the final kill stage. Historical prices are assumed. Requires accounts at several books. |
| 2 | **Unders on high totals.** | Actual scoring rises only ~0.9 points per posted point, and the shortfall is in the second half. 54.3% at totals ≥ 60 (2017–25, n = 1,523). | Confidence interval (51.8%–56.8%) still contains break-even. Profitable only at −112 or better; loses at −120. Overtime reliably pushes games over. |
| 3 | **Open-field-yards underdog.** Back the dog when the favorite plays in games with more long gains. | Reported at 56.2% (2018–25). | Rebuilt from scratch it reproduces at only ~53.4% (z < 1). Weakest of the three. |

## Execution rules (not edges)

- **Shop every bet.** The best available number beats the median by about
  +1.0 ROI points on spreads and +2.9 to +4.0 on moneylines. It turns a losing
  bettor into a slightly-less-losing one; it cannot create an edge.
- **Buy the half point off 7.** Games land exactly on 7 about 9% of the time,
  against ~4% needed to justify the usual 10-cent charge.

## What the paper ledger tracks

Each week the ledger records the best available number for each candidate,
captures the closing number before kickoff, and grades the result.

- **Watch closing line value (CLV), not win rate.** At ~100–150 bets a season,
  a win rate is too noisy to mean anything; beating the closing number shows up
  in weeks.
- Unders are only logged at −112 or better when real prices are available.
- Bets priced from the free data source carry an **assumed** −110. Their ROI is
  not real until a live odds feed is connected.

## What would change the answer

1. **A live odds feed with real prices, 10+ books.** Required to execute
   middles, enforce the unders price limit, and measure CLV.
2. **Injury and availability data with timestamps.** The one information class
   nobody could test; the market overcharges for QB instability by ~0.8 points
   against ~0.9 needed to clear the vig.
3. **Weather**, as a control. Wind is the only real physical mechanism for
   totals, and it may *explain away* the unders rather than strengthen them.

Decision rule, written in advance: nothing gets real money in 2027 unless a
candidate shows **sustained positive CLV on real prices** across the 2026 season.
