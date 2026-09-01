# Line shopping + paper trading

## What this is for

Line shopping is the only mechanical, repeatable edge the whole search turned
up. It is also **necessary but not sufficient**: measured on 5,620 games it
moves an always-bet-home strategy from −4.40% to −3.40% ROI. Better. Still
losing. It multiplies an edge; it cannot create one.

So the goal for 2026 is **evidence, not profit**: log picks, capture closing
lines, and measure CLV. Nothing here risks money.

## Why a real odds feed matters

Everything measured on CFBD is a **lower bound**, because CFBD carries only 3
books (DraftKings, ESPN Bet, Bovada) and typically returns 2 per game.

That thinness breaks the maths in a specific way. The "fair" line is the median
across books — with 2 books that is just their midpoint, so the better book is
*mechanically* half the gap better than "consensus" every single time. Run the
CFBD fallback and you will see +EV on **both sides** of most games. That is not
free money, it is an artifact. `wf/shop.py` warns when mean books < 4 and flags
`both_sides_pos` rows for exactly this reason.

Measured shopping value on CFBD's books:

| market | value |
|---|---|
| spread, all games | +1.00 ROI pts |
| spread, when books disagree by 2+ (4% of games) | +10.59 ROI pts |
| moneyline | +2.9 to +4.0 ROI pts |

Moneylines are where books differentiate most — the best home price beats the
median in 87.2% of games. **Shop moneylines before spreads.**

## Setup: The Odds API

1. Get a free key at <https://the-odds-api.com> (500 requests/month, which is
   plenty for weekly polling — one call returns the whole board).
2. Add it to `.env` alongside the CFBD key:

   ```
   CFBD_API_KEY=...
   ODDS_API_KEY=...
   ```

   `.env` is already gitignored.
3. Verify:

   ```bash
   python -m wf.shop 2026 1
   ```

   With a key you should see 10-15 books and no thin-consensus warning. Without
   one it falls back to CFBD and says so.

Quota note: `fetch_theoddsapi` prints remaining requests from the response
headers. One call per week per market set is the intended cadence.

## Weekly workflow

```bash
python -m wf.paper log   2026 1     # Tue: record picks at the best number now
python -m wf.paper close 2026 1     # Sat pre-kickoff: capture closing lines
python -m wf.paper grade 2026       # Sun: settle against final scores
python -m wf.paper report 2026      # CLV + running results
```

The ledger is `paper_ledger.csv` at the repo root.

## Reading the output

**CLV is the number to watch.** Results need hundreds of bets before a win rate
means anything — at ~160 bets a season, one season of unders has a standard
error of ~3.9 points, so a 54% record and a 48% record are both consistent with
having no edge at all. CLV converges far faster.

- `clv_points` — how many points better than the closing number you got.
  Positive is good.
- Beating the close **consistently** is the precondition for winning long term.
  Negative CLV plus a winning record means you got lucky, not sharp.

## Strategies logged

Deliberately short — only what survived the sweep (see
`memory/edge-sweep-results.md`):

- `high_total_under` — the one live signal. 54.4% historically at totals ≥60,
  monotone across thresholds, 7/8 seasons positive, and stable out-of-sample.
  But the 95% CI is [52.10%, 57.51%], which **contains** the 52.38% break-even.
  This is being tracked to gather evidence, not because it is proven.
- `shop_value` — offers where the best book beats the median by ≥1% EV. Tests
  the shopping thesis directly.

Do not add situational angles here. Forty of them were tested and none survived
FDR correction.

## How offers are priced

Naively taking the biggest number is wrong: +7 at −130 can be worse than +6.5
at −105. So each offer is priced against the no-vig consensus using the
empirical CFB residual distribution in `margin_dist.json` (5,605 games,
2018-2025):

- spread residual: sd 15.45, mean −0.01 (market is unbiased)
- total residual: sd 16.04, mean **+0.37** (slight under lean — consistent with
  the unders finding)

Half-point values come straight out of it, and match the raw data:

| line moved in your favour | cover rate gain |
|---|---|
| 0.5 | +1.89 pp |
| 1.0 | +3.32 pp |
| 3.0 | +8.46 pp |

Useful framing: **~1.5 points of line value is worth the entire vig.**

Pushes are handled explicitly — treating them as losses understates EV at
integer lines, where the point mass is real (residual lands exactly on 3 in
1.16% of games, on 7 in 0.98%).

## Gotcha: never average American odds

The scale is discontinuous at ±100, so `median(+180, −110)` returns 35, which is
not a price. This produced impossible output (a *better* price showing a
*worse* return) before it was caught. Use `dataset.american_median` /
`american_best`, which convert to decimal, aggregate, and convert back.
