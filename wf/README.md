# `wf/` — walk-forward harness

A replacement for the old backtest path, built after the 2025 models turned out
to be reporting a leaked 58.7% ATS (see `archive/2025_leaked/README.md`).

The point of this package is narrow: **make it hard to fool yourself.** It is
not a model that beats the market. It is the apparatus that tells you whether
something does.

## The as-of-week contract

For a game in season `S`, week `W`, a feature may only use information that
existed before kickoff. Three CFBD endpoints look safe and are not — all three
were verified empirically on 2026-08-04, not assumed:

| Endpoint | Naive use | Reality | Safe use |
|---|---|---|---|
| `/ratings/elo?year=S&week=W` | rating going into week W | **contains week W's result** — corr(elo[W]−elo[W−1], week-W margin) = **+0.824** | `week = W-1`; `week=0` is preseason |
| `/stats/season/advanced?startWeek=1&endWeek=W` | stats through week W−1 | **`endWeek` is inclusive** — Georgia played week 1 and `endWeek=1` already shows 76 plays | `endWeek = W-1` |
| `/ratings/srs?year=S`, `/ratings/sp?year=S` | in-season power rating | **full-season aggregate**, computed from results including this game | season `S-1` only |

Safe without qualification: `/talent` and `/player/returning` (published
preseason), betting lines, and anything schedule-derived (rest, neutral site).

`wf/dataset.py` implements this in one line per row — `asof = w - 1` — and every
feature family is tagged in its `PROVENANCE` dict.

Week-1 games therefore carry **no** current-season form features. They are NaN,
on purpose, and the models handle NaN natively rather than imputing a number
that was never knowable.

## The audit

```bash
python -m wf.audit            # defaults to 2023-2025
python -m wf.audit 2019 2020 2021 2022 2023 2024 2025
```

It asserts the structural invariants (as-of window strictly before the game,
week-1 rows empty, cumulative play counts monotone, spread sign convention) and
then runs the canary:

> A closing spread is a very efficient forecast. After you subtract it, what's
> left (`ats_diff`) should be close to noise. Any pre-kickoff feature with a
> large correlation against `ats_diff` is far more likely to have seen the
> result than to be worth that much edge.

Anything over |0.10| gets flagged. For calibration, the audit also prints the
known-bad feature side by side with its safe twin:

```
same-season srs_diff  vs ats_diff = +0.256   <-- leaked
prior-season srs_diff vs ats_diff = +0.003   <-- safe
```

That gap is the entire mirage. Run this after any change to `dataset.py`.

## The backtest

```bash
python -m wf.backtest 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
```

Trains a fresh model for every `(season, week)` on games that finished strictly
earlier, so there is no fitting on the future and no single train/test split to
overfit to.

It models **margin** and **total points** as regressions rather than classifying
"cover", because the quantity that matters is how far the model's number sits
from the market's. That gap is the edge, and results are reported across
increasing edge thresholds.

Read the **`z`** column, not `win_pct`. It measures distance from the -110
break-even of 52.38%. `|z| < 2` is noise, and since several thresholds are
tested at once you should want considerably more than that before risking money.
The report also prints no-model baselines (always home, always favorite, always
over) and the model's MAE against the market's own MAE — if the model can't
forecast the raw number better than the closing line, it has no informational
edge to convert.

## Layout

- `cache.py` — disk-cached CFBD fetch (`data_cache/`). Historical data is
  immutable, so first run is slow and every run after is instant.
- `dataset.py` — the as-of-week feature table. **All leakage rules live here.**
- `audit.py` — invariants + the leak canary. Run after touching `dataset.py`.
- `backtest.py` — weekly-refit walk-forward, betting metrics, baselines.
