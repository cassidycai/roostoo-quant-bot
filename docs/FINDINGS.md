# Findings — what the backtest actually says

Everything below comes from real Binance 30-minute candles over **120 days**
(2026-06-28 → 2026-09-28), 8 majors (BTC, ETH, SOL, BNB, XRP, DOGE, LINK,
AVAX), with the competition's cost model: **0.1% taker per side + 5 bps
slippage**, market orders filled at the **open of the next bar**.

Reproduce:

```bash
python scripts/fetch_history.py --days 120 --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,LINKUSDT,AVAXUSDT
python run_backtest.py --oos-frac 0.25
python scripts/analyze_backtest.py reports/trades_in-sample.csv
python run_sweep.py --config-grid "@reports/grid_stop.json"
python run_sweep.py --grid "@reports/grid_direction.json" --config-grid "@reports/grid_exit.json"
```

> **Configuration note.** Sections 1–3 were produced with **Rule 4's deviation
> gate enabled** (`enforce_min_deviation: true`), which is how those tables were
> obtained. That gate belongs to another owner, so it now ships **disabled** and
> the shipped default is worse on its own: in-sample −18.6% rather than −13.4%
> (section 4 has the full comparison, and it is the same conclusion either way).
> To reproduce these exact figures, add `"enforce_min_deviation": true` to
> `STRATEGY_PARAMS`, or pass `--params '{"enforce_min_deviation": true}'`.
> On Windows/PowerShell, inline JSON is mangled by the shell — use the `@file`
> form instead.
>
> **These numbers have moved once already.** Section 1 was regenerated after a
> correctness pass over the backtester (fill timestamps, the out-of-sample
> measurement window, and refusing an entry whose stop cannot be computed). The
> remaining parameter grids in sections 2 and 4 still carry the pre-pass values;
> their *ordering* is what they were written to establish, and that is unchanged,
> but re-run `run_sweep.py` before quoting an individual cell as current.

**Headline: as specified, Rules 1–3 lose money — and the entry logic is not the
reason.** Two configuration choices cause almost all of the damage, and both are
owned by teammates (Rules 5 and the short leg of Rule 2).

## 1. The default configuration loses

| | in-sample (90d) | out-of-sample (30d) |
|---|---|---|
| total return | −13.4% | −11.9% |
| Sharpe | −8.1 | ≤ −10.0 |
| Sortino | −8.5 | ≤ −10.0 |
| Calmar | −3.3 | −6.6 |
| composite | −6.8 | −9.0 |
| round trips | 333 | 144 |
| **fees paid** | **9,121** | **4,055** |

> These figures were regenerated after the correctness pass that fixed the
> backtest's fill timestamps, restricted the out-of-sample metrics to the
> post-split window, and made an uncomputable stop refuse the entry. The
> conclusion is unchanged; the last decimal is not.

`≤ −10.0` marks a **clamped** figure, not a measurement. Ratios are capped at ±10
for reporting, and both out-of-sample ratios reach the cap — the raw values are
−14.5 (Sharpe) and −11.8 (Sortino), so the composite of −9.0 is *better* than the
published 0.4/0.3/0.3 formula applied to the measured ratios (which gives −11.0).
Treat every capped number in this document as a lower bound, and note that the
cap is saturation rather than scaling: it compresses every configuration whose
true Calmar is below −10 onto the same value, which is part of why the
out-of-sample composites in section 5 cluster so tightly.

Fees alone are 9.1% of NAV. Gross P&L is negative too (−4,329 across the round
trips), so cost is not the only problem.

## 2. The Rule 5 stop is where the money goes

`scripts/analyze_backtest.py` splits P&L by exit reason, and the picture is
unambiguous:

| exit | n | gross P&L | avg bars held | win rate |
|---|---|---|---|---|
| z-exit (Rule 2) | 86 | **+12,176** | 6.0 | **100%** |
| time stop (Rule 6) | 102 | +2,955 | 12.0 | 49.0% |
| **stop (Rule 5)** | 145 | **−19,460** | 4.2 | **0%** |

Every single trade that reached the mean-reversion target won. The ATR stop —
hit after an average of **4.2 bars** — converted 145 of them into realised losses
worth more than everything else earned combined.

Widening it improves the score monotonically, which is the signature of a stop
that sits inside the noise:

| `stop_atr_mult` | IS composite | OOS composite | IS return |
|---|---|---|---|
| `null` (disabled) | **−4.85** | **−7.86** | −11% |
| 5.0 | −5.02 | −8.77 | −10% |
| 3.0 | −6.08 | −8.91 | −14% |
| 1.5 (default) | **−7.13** | **−8.94** | −14% |

> This grid was produced with the deviation gate enabled, before the correctness
> pass, so the 1.5 row reads −7.13 where the regenerated full-sample run gives
> −6.81. The ordering — which is the whole point of the table — is unaffected.

### Why Rules 2 and 5 are in direct conflict

This is structural, not a tuning accident. Rule 2 deliberately buys an asset that
has just moved hard *against* it (`Z ≤ −2`). Rule 5 then exits if the adverse move
exceeds `1.5 × ATR(14)` — and ATR is itself elevated *because of the very move
that triggered the entry*. The entry signal and the stop trigger are close
together by construction, so the stop fires while the deviation is still
widening, before there has been any time to revert.

A coherent alternative is to stop on the **same signal that generated the
entry**: exit if `Z` extends beyond, say, `−3.5` (the deviation failed to
revert and is now more extreme), rather than on an unrelated ATR multiple. That
keeps one author for the entry and the exit condition.

**Related code change.** Because the measured risk budget depends on the stop
distance, an entry whose stop distance cannot be computed (a missing or
non-finite ATR) used to be sized on the caps alone — i.e. *larger* than Rule 7
permits, with no stop. That is now refused by default:
`REFUSE_ENTRY_WITHOUT_STOP=1`. It does not disturb the `null` rows above, because
turning `STOP_ATR_MULT` off is a deliberate configuration choice rather than a
computation failure, and the fallback still applies there.

## 3. The short leg is the single biggest destroyer

Sweeping direction together with the time stop (stop disabled, to isolate them):

| shorts | time stop | IS composite | OOS composite | **IS return** | **OOS return** |
|---|---|---|---|---|---|
| allowed | 12 bars | −4.85 | −7.86 | −11% | −12% |
| allowed | off | −4.31 | −6.75 | −20% | −14% |
| **disabled** | 12 bars | −0.76 | −7.29 | **−1%** | −5% |
| **disabled** | off | **−0.60** | **−6.19** | **−0.6%** | **−4%** |

Turning the short leg off takes the in-sample result from −11% to **−0.6%** —
essentially flat before costs. By direction, in-sample gross P&L is −46 for longs
and −4,283 for shorts: the long book is already break-even, and essentially all
of the gross loss comes from fading upward stretches.

The plausible reason is drift. Over this sample crypto rose, so shorting a
`+2σ` stretch is fighting a persistent trend, while the equivalent long trade is
fading *into* it.

## 4. Rule 3 (ADX) and Rule 4 (deviation) both earn their place

Full z-entry × ADX × deviation grid, in-sample return:

| `z_entry` | ADX filter | dev gate | IS return | OOS return | fills (IS/OOS) |
|---|---|---|---|---|---|
| 2.0 | on | on | −14.3% | −12.0% | 676 / 298 |
| 2.0 | **off** | on | −19.5% | −20.2% | 882 / 606 |
| 2.5 | on | on | −10.4% | −7.6% | 400 / 180 |
| 2.5 | **off** | on | −15.9% | −16.1% | 1070 / 433 |
| 3.0 | on | on | −5.6% | −5.4% | 220 / 98 |
| 3.0 | **off** | on | −7.1% | −8.3% | 560 / 251 |

* **The ADX filter helps consistently**, at every z-entry level, by refusing to
  fade genuine trends. It roughly halves the trade count and cuts the loss.
* **The deviation gate (Rule 4) helps modestly** at every level.
* **Higher z-entry is better**: `3.0` loses about a third as much as `2.0`,
  because it trades far less often against a fixed 0.2% round-trip cost.

All 12 cells still lose, so no parameter choice rescues the current exit policy.

## 5. The out-of-sample result does not generalise

Every in-sample improvement collapses out-of-sample: OOS composites cluster in a
narrow band (−8.4 … −9.7) no matter how the parameters are set. **A tight OOS
cluster around a loss means the problem is structural, not parametric** — there is
no plateau to find. Any tuning that only looks good in-sample is noise.

Two cautions on reading that band. It is bounded below by the ±10 reporting clamp,
which saturates rather than scales, so configurations that differ in reality can
print the same composite. And the OOS window in those runs is only **30 days**
(1,441 bars of the 5,761), because that is how long the dataset was, which is too
short to conclude anything about regime robustness. `data/` now holds **365 days
for all ten tradable pairs**, so re-run it and check that the result holds in both
a rising and a falling market before committing.

## 6. What to change first, in order of expected impact

1. **`allow_short: false`** — worth roughly +10 percentage points in-sample.
2. **`STOP_ATR_MULT=none`** (or ≥ 3× ATR, or a structural `Z`-based stop) — worth
   roughly +2 to +3 points, and it is the difference between a 0% and a ~40% win
   rate on stopped trades.
3. **`z_entry: 2.5` or `3.0`** — fewer, more selective trades, materially smaller
   losses at a fixed 0.2% round-trip cost.
4. Keep **ADX < 25** and the **0.6% deviation gate**.
5. Re-examine the **12-bar time stop**: it is roughly neutral in-sample and
   slightly harmful out-of-sample; the 90 time-stop trades averaged exactly 12.0
   bars, so the cap is binding often. Try 18–24 bars.
6. Then test the **supplementary filters** one at a time (see `run_sweep.py`).
   Their maths is implemented and unit-tested; all four are off by default.

Even after all of that the best cell is **−4% out-of-sample**, so the honest
expectation is that this needs a genuine edge improvement, not just tuning. The
one encouraging number is that the reversion target itself is reached reliably:
when a trade is allowed to reach the mean, it wins 100% of the time. The problem
is how many trades get killed before they get there.

## 6b. The edge itself, measured — and why tuning cannot fix it

Everything above reports what the *strategy* did, which mixes an edge, an exit
policy and a cost model. `scripts/edge_analysis.py` separates them, and the
separation is the important result:

```bash
python scripts/edge_analysis.py                       # the shipped setup
python scripts/edge_analysis.py --folds 5             # the decisive stability test
python scripts/edge_analysis.py --z-entry 3.0 --gate 0.006
python scripts/edge_analysis.py --data-dir data_4h --interval 4h --match-span
```

**The mean does revert.** Of 570 Rule 2 long entries (ADX < 25), Z reached the
−0.25 exit target within 24 bars **64.9%** of the time, median 9 bars (~4.5h). So
the signal is not noise, and the 100% win rate on z-exits is not an artefact.

**The edge is smaller than the fee.** Measured with no stop at all, so the exit
policy cannot be blamed, and filling one bar after the signal — which is what
`Backtester` actually does (`execution_delay_bars=1`):

| | n | mean gross | median gross | net of 0.30% |
|---|---|---|---|---|
| hold to the −0.25 target | 570 | **−0.036%** | −0.010% | **−0.336%** |
| the same at maker cost (0.25%) | 570 | −0.036% | −0.010% | −0.286% |

A 2σ deviation reverting to −0.25σ is a move of about **0.1–0.3%**, and a round
trip costs **0.30%**. The target distance and the cost are the same order of
magnitude, so a completed trade is a coin flip on a 0.3% fee. **No stop parameter,
no sizing rule and no filter can rescue that**, because none of them change what a
completed round trip is worth.

> **A measurement trap worth recording.** Filling at the signal bar's own close
> instead of the next bar's gives +0.095% where the honest figure is −0.036%. That
> one-bar difference is 0.13 percentage points — four times the entire edge being
> measured, and enough to flip the conclusion. Any measurement of this strategy
> must state its fill assumption.

**No parameter survives the decisive test.** Split the timeline into five
consecutive folds and require the sign to hold; a single 60/40 split can be passed
by luck, and a grid can be mined until one cell looks good in both halves. Net of
0.30%:

| z_entry | fold 1 | fold 2 | fold 3 | fold 4 | **fold 5 (most recent)** |
|---|---|---|---|---|---|
| 2.0 (shipped) | −0.756% | −0.056% | −0.334% | −0.331% | **−0.240%** |
| 2.5 | −0.548% | −0.127% | −0.378% | −0.370% | **−0.241%** |
| 3.0 | −0.306% | −0.062% | −0.335% | −0.442% | **−0.219%** |
| 3.5 | −0.165% | +0.221% | −0.318% | −0.791% | **−0.292%** |

Every threshold loses in four of five folds and loses in the most recent one. The
single positive cell (z ≥ 3.5, fold 2) sits between two negative folds, which is
what noise looks like. **This is the mathematical form of "there is no parameter
plateau"** in section 5: the folds disagree with each other, not with the
parameter, which means the outcome is driven by which kind of market happened to
be in the window.

**Raising the threshold does not clear the cost either.** Selectivity cuts the
*number* of trades without raising the per-trade edge above the fee, and the
thresholds that come closest to it are the ones that barely trade:

| z_entry | entries/day | over a 14-day window |
|---|---|---|
| 2.0 | 4.75 | 67 |
| 3.0 | 0.95 | 13 |
| 3.5 | 0.47 | **7** |
| 4.0 | 0.18 | **3** |

**A longer bar does not fix it either.** On 4-hour bars (2 years, 8 majors) the
same measurement gives 0.04–0.15 trades per day — 1 to 2 trades in a 14-day
window — so there is nothing to average, whatever the per-trade number says.

**The Rule 4 deviation gate is not a fix.** The team runs with
`enforce_min_deviation: true`. Measured with the gate off versus on, on the same
data and the same cost model, it moves the in-sample composite from −7.93 to
−6.81 and cuts fills by ~20%, but the loss per trade is unchanged: it removes
trades broadly rather than removing the unprofitable ones. It is not harmful and
it is worth keeping; it is not the lever.

**Conclusion.** The strategy is not `stop_atr_mult` away from working. It is
structurally unprofitable at this horizon and this cost, which is why section 5's
grid loses in all 12 cells and why every in-sample improvement died out of sample.
What would change the answer is a signal whose *typical completed move* is several
times the 0.30% round trip — a materially longer holding period, or a genuine
regime filter that refuses to trade when the reversion does not show up (see the
fold table: folds 1, 3 and 4 are where the money goes). Not another pass over
these parameters.

### 6b.1 Maker execution, measured

The bot sends market orders only (`engine.py` uses `place_order(..., "MARKET")`),
so it pays the taker fee *and* crosses the spread on every entry. A resting bid at
the signal bar's close pays the maker fee and crosses nothing. The tool models it:

```bash
python scripts/edge_analysis.py --maker
```

The fill assumption is the whole argument, so both readings are reported.
`touch` fills when the fill bar's low reaches the limit (first in the queue);
`through` demands the bar trade strictly *below* it (pessimistic). An unfilled
order means the trade does not happen at all — fewer trades, not the same trades
cheaper.

| entry | n | filled | mean net | win |
|---|---|---|---|---|
| market order (what ships today) | 570 | 100% | −0.336% | 28.8% |
| limit, `touch` | 563 | **98.8%** | −0.285% | 32.5% |
| limit, `through` | 543 | **95.3%** | **−0.289%** | 32.4% |

Two things worth stating plainly:

* **It helps, by about a seventh of the loss.** −0.336% → −0.289% is **+0.047 per
  trade**. The saving is smaller than it first looks because only the *entry* is
  passive — the exit still crosses the spread, so the round trip goes 0.30% →
  0.25% (0.1% taker + 0.05% maker + slippage on both sides), not down to 0.15%.
* **It does not fix anything.** All five folds stay negative, including the most
  recent (−0.196%), so it fails the same test the parameters fail.

The **95–99% fill rate is the softest number in this document** and should not be
trusted as much as the rest of it. It comes from OHLC bars, and "the low reached my
limit" is not the same as "my order was at the front of the queue at that price".
In a real book a resting bid fills only when someone sells into it, and queue
position delays or loses the fill. Treat `through` as the number to plan on, and
confirm the real rate in the live journal before relying on it.

> **How to check any future change with this.** Run `edge_analysis.py`, read the
> `net @ 0.30%` line in part 2, and require part 4 to be positive in most folds
> and not negative in fold 5. A change that only improves the full-sample total
> return has not been demonstrated to work; it has been fitted.

## 7. Method caveats (read before quoting any of this)

* **Removing the stop also removes risk-based sizing.** With no stop distance,
  Rule 7's 0.5% risk budget cannot be expressed, so `PositionSizer` falls back to
  the 15% per-pair cap and positions get *larger*. The `stop_atr_mult: null` rows
  therefore change two things at once. The clean version of that experiment is
  `STOP_ATR_MULT=none` **plus** `STOP_LOSS_PCT=0.05`, which keeps sizing
  risk-based with a wide stop.
* **Rule 1's spread filter cannot discriminate in a backtest.** The CSV has no
  book, so one assumed spread (`--spread-bps`, default 5) is applied to every
  pair. The ranking and the code path are exercised; the *filter* is not
  validated. Depth is not checked at all offline.
* **Bars are sampled, not true OHLC, on the live venue.** Live bars come from
  polling the ticker, so an intrabar spike between two samples is invisible. The
  backtest uses real Binance OHLC, so live results will differ from these.
* **Gap risk through a stop is understated.** A stop fills at its own level plus
  slippage; a genuine gap through it would be worse.
* **`stdev` uses the sample (n−1) denominator**, and ADX follows Wilder's
  original seeding. Both are defensible conventions, but a different choice moves
  the last decimal of a z-score and can flip a trade right at the threshold.
* **The sweep's `OOS/IS ratio` flag is suppressed when |IS composite| < 0.5**, so
  it will print `n/a` for near-zero baselines instead of reporting a meaningless
  ratio like "10.34, ok".
* **The venue tracks Binance — confirmed by the organisers.** That is the premise
  that makes Binance-derived signals valid here, and it is now measured rather than
  assumed: `roostoo/basis.py` compares every quoted pair against Binance each bar,
  journals the readings, and excludes any pair beyond `BASIS_MAX_PCT` (default 1%).
  The seeded-history tolerance in `engine.seed_history` was tightened from 2% to
  0.5% for the same reason — a gap that large is a wrong symbol (USDT vs USD) or a
  stale feed, not a related market. Note this check **fails open** (a Binance
  outage must not stop trading), unlike the depth check, which fails closed.

## 9. Direction: momentum beats reversion

Rule 2's entry can be read two ways, and the code now makes the choice explicit
through `direction` instead of leaving it implicit in a sign.

**`data/` now carries 365 days for every tradable pair, which is enough to re-run
this at one year.** The table below was produced from a 2-year dataset (35,042 x
30m bars, 10 pairs) and holds out the final 183 days, so the out-of-sample column
is comparable across rows. Market data is git-ignored, so that particular set is
not in the repository; `scripts/fetch_history.py --days 730` regenerates an
equivalent one.

| policy | IS | OOS (183 days) | IS drawdown | round trips (IS/OOS) |
|---|---|---|---|---|
| reversion, shipped defaults | -19.65% | -13.72% | 19.85% | 315 / 498 |
| reversion + the team's `.env` | -19.85% | **-19.25%** | 19.97% | 488 / 317 |
| **momentum + the team's `.env`** | -19.32% | **-2.51%** | 19.71% | 310 / 145 |
| **momentum + stricter entry params** | +2.65% | **-1.51%** | **3.05%** | 57 / 31 |

`run_backtest.py --data-dir <2yr> --oos-frac 0.25`.

**Momentum beats reversion by 11-17 percentage points out of sample**, with about
a fifth of the drawdown, and stricter entry parameters (`z_entry 2.8`,
`adx_max 20`, `min_abs_deviation 0.012`) are worth roughly another 1pp against
the looser `.env` values.

Three things this does **not** say:

* **It is not profitable.** The best out-of-sample cell is -1.51% over 183 days.
  Momentum turns a large loss into a small one; it does not produce a gain.
* **It trades too little to comply.** That best cell makes 31 round trips in 183
  days -- about 2 in a 14-day window, against a requirement of 8 active trading
  days. The configurations that trade enough (145-317 round trips) lose more.
* **Magnitudes do not transfer between datasets.** On the repo's own 120-day data
  this same comparison reads -0.55% vs -4.22%; on 2 years it reads -2.51% vs
  -19.25%. Only the *ordering* is stable. That is why the team needs one shared,
  fingerprinted dataset: `data/*.csv` is gitignored, so every member currently
  backtests against a different history. A single month of this strategy was
  measured at +4.30% while the two-year out-of-sample result is -1.51%.

Two mechanics worth keeping:

* **The entry and the exit must move together.** A momentum long enters at
  `Z >= +2.8` and can only leave through the Z-exit when `Z <= -0.25` -- a swing
  of more than three sigma through the entire distribution. Ship one without the
  other and the Z-exit silently never fires; positions die on the stop instead.
  That is why `direction` drives both.
* The ATR stop **helps** momentum (-0.55% OOS with it, -1.16% without, on the
  120-day set) while it **hurt** reversion (section 2). The sign of that result
  depends on the direction, so an exit finding must not be carried across.
