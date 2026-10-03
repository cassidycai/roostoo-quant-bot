# Roostoo Quant Trading Hackathon - autonomous trading agent

An autonomous, dependency-free Python trading bot for the **HK vs AU vs IN Quant
Trading Hackathon** on [Roostoo](https://luma.com/coghwiyt)'s mock crypto
exchange, together with the backtesting and diagnostic tooling needed to tune it.

The competition scores a portfolio as `0.40 x Sortino + 0.30 x Sharpe + 0.30 x
Calmar`. The design is therefore **risk-first**: downside deviation and drawdown
are the objective, not an afterthought. Position sizing, the exit order and the
kill switch are all written to bound the loss before they are written to seek a
gain.

```bash
python run_live.py --check                 # verify keys/clock/venue (read-only, no orders)
python run_live.py --mock --cycles 20      # full loop against the offline simulator
python scripts/fetch_history.py --days 365 --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT
python run_backtest.py --oos-frac 0.25     # in-sample vs out-of-sample
python scripts/analyze_backtest.py reports/trades_out-of-sample.csv   # why did it lose?
python run_sweep.py --config-grid "@reports/grid_stop.json"           # tune one decision at a time
```

> **Read [`docs/FINDINGS.md`](docs/FINDINGS.md) before trusting the default
> parameters.** It records every ablation we ran, including the ones that failed.
> The strategy is **not profitable** on any window we have measured; section 10
> states that plainly.

## Submission materials

Everything submitted is **Markdown**, and nothing else needs to be read:

| File | Contents |
|---|---|
| `README.md` | this document - portfolio, strategy, engine, backtest, fees, risk |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | the full research record: measurements, ablations, negative results |
| [`docs/SECURITY.md`](docs/SECURITY.md) | credential handling and incident response |
| [`deploy/AWS_DEPLOY.md`](deploy/AWS_DEPLOY.md) | the live deployment we run |

No spreadsheets, slide decks or PDFs are part of the submission. Every number in
this document is reproducible with the commands in section 9.

---

## 1. How the portfolio is curated

Five decisions, in the order the engine makes them. Each one is a section below.

| # | Decision | Where | Section |
|---|---|---|---|
| 1 | **Which coins may we hold?** Rule 1: rank by 24h turnover, apply a spread ceiling and a depth screen | `universe.py` | 6, 7 |
| 2 | **When do we buy?** Rule 2 z-score, confirmed by Rule 3 (ADX) and Rule 4 (deviation must clear costs) | `strategies/mean_reversion.py` | 2 |
| 3 | **How much?** Rule 7's 0.5% NAV loss budget per trade, sized from the actual stop distance | `risk.PositionSizer` | 6 |
| 4 | **When do we sell?** Rule 2's mean exit, Rule 5's ATR stop, Rule 6's time stop | `risk.protective_exits` | 6 |
| 5 | **What does it cost?** 0.1% taker per side plus slippage, priced into every fill and into the entry gate itself | `backtest.py`, `strategies/scoring.py` | 5 |

The portfolio is deliberately **concentrated but capped**: at most 4 positions
(`MAX_OPEN_POSITIONS`), at most 15% of NAV in any one coin (`MAX_PAIR_WEIGHT`) and
at most 60% of NAV gross (`MAX_GROSS_EXPOSURE`). `0.15 x 4 == 0.60`, so
equal-weight sizing across four slots satisfies Rules 8, 9 and 10 simultaneously.
The remaining 40% stays in cash, which is what keeps the drawdown term of the
score small - and we consider that a feature, not idle capital.

## 2. Strategy: what it does, and how we arrived at it

### 2.1 The Rule 2 signal

```
SMA48 = mean of the last 48 closes          (48 x 30min = 24h)
Std48 = standard deviation of those closes
Z     = (Close - SMA48) / Std48
```

Rule 3 admits an entry only when `ADX(14) < 25`, i.e. when the market is ranging
rather than trending. Rule 4 adds an economic gate: `|Close - SMA48| / Close` must
exceed 0.6%, so we do not pay a 0.2% round trip to capture a 0.05% move.

### 2.2 The direction question, which the rulebook states ambiguously

The rulebook's long clause reads *"Z[t-1] >= X and Delta Z < 0"*. That sentence
mixes a **momentum level** with a **reversion confirmation sign** - a long entry
conventionally needs the price *below* the mean, and `Z >= +2` is the opposite.
The text is genuinely ambiguous, so we made the choice explicit in code
(`direction`) and then measured both readings instead of arguing about it.

Held-out comparison, 2-year dataset, 10 pairs, final 183 days held out so the
out-of-sample column is comparable across rows:

| policy | in-sample | **out-of-sample (183 days)** | in-sample drawdown | round trips (IS/OOS) |
|---|---|---|---|---|
| reversion, shipped defaults | -19.65% | -13.72% | 19.85% | 315 / 498 |
| reversion + our `.env` | -19.85% | **-19.25%** | 19.97% | 488 / 317 |
| **momentum + our `.env` (shipped)** | -19.32% | **-2.51%** | 19.71% | 310 / 145 |
| momentum + stricter entry params | +2.65% | **-1.51%** | **3.05%** | 57 / 31 |

**Entering with the stretch beat fading it by 11 to 17 percentage points out of
sample, at about a fifth of the drawdown.** That is why the shipped policy is
`direction: momentum`: we buy a coin that is stretched *up*, and short one that is
stretched *down*.

The stricter parameter set in the bottom row has a far better drawdown but only 31
round trips out of sample, which we judge too few to satisfy the competition's
8-active-day requirement, so we ship the middle row deliberately.

### 2.3 What we are honest about

* The strategy is **not profitable** on any window measured. Momentum converts a
  large loss into a small one; it does not create a gain.
* Magnitudes do not transfer between datasets. On the 120 days committed to this
  repository the same comparison reads -0.55% vs -4.22%; on 2 years it reads
  -2.51% vs -19.25%. Only the *ordering* is stable, so we report both.
* The window trends upward, so part of the momentum result may be that regime
  rather than a durable edge. Sections 8 and 10 say more.

## 3. The trading engine

`roostoo/engine.py` is the autonomous loop. Four constraints shaped it:

1. **Rules 2-3 are defined on closed 30-minute bars**, so the strategy runs
   exactly once per bar - not once per loop.
2. **Rule 5's stop is a price level**, so protective exits are checked on *every*
   loop, not only at the bar boundary.
3. **Exits must be more reliable than entries.** Flattening has to work even when
   the risk layer is halting new business.
4. **A 14-day unattended run will restart.** Position state is persisted, and
   every cycle reconciles the local book against the exchange's balances - the
   exchange is the source of truth for quantity.

**Live and backtest are the same code.** The backtester reuses `RiskManager`,
`PositionSizer`, `PositionBook` and the same `Strategy.generate` call rather than
reimplementing them, so a difference between the two is a bug rather than a
modelling choice. Both evaluate the strategy exactly once per closed bar and check
stops on every iteration.

**Order handling is designed around the failure that costs the most.** A transport
failure leaves an order genuinely unknown, so `place_order` is never blind-retried:
the client returns `status="UNKNOWN"` and the engine reconciles against the order
history. A history row is only accepted as ours when it is genuinely `FILLED`,
matches the side, matches the quantity to within one lot, and reports something
filled - matching on side and recency alone once accepted a cancelled order of an
unrelated size.

**More failure modes we closed**, each of which has a regression test:

* An absent balance row is not a zero balance. Only an explicit zero closes a
  position; a truncated payload used to read as "the exchange holds nothing" and
  delete the whole book.
* A failed short-position query is not "no shorts". The call now distinguishes
  "unknown" from "none", because an empty result once deleted every short along
  with its collateral and its stop.
* A stop that cannot be computed refuses the trade. With `STOP_ATR_MULT` set, a
  non-finite ATR no longer approves a caps-only position with no stop at all.
* A position is always closable: the time stop is evaluated before the mark is
  validated, and the execution layer falls back to the live quote and then the
  entry price, so a holding whose mark is zero can still be exited.
* A partial balance snapshot is journalled and skipped rather than priced at zero,
  which would otherwise trip the permanent kill switch.

Every decision, rejection and reconciliation is appended to a JSONL journal plus a
trades CSV, so any live result can be reconstructed after the fact.

## 4. Backtesting

```bash
python run_backtest.py --data-dir data --oos-frac 0.25
```

| Property | How it is handled |
|---|---|
| Sample split | the first 75% of the timeline is in-sample, the last 25% out-of-sample |
| Warm-up | the out-of-sample run is warmed with the bars *before* the split, so the two halves are comparable |
| Look-ahead | an order decided on bar *t* fills at the **open of bar t+1** - the signal needs bar *t*'s close to exist, and that price is gone by the time an order can be sent |
| Stops | detected against the bar's high/low and filled **at the stop level plus slippage**, which still understates a real gap |
| Fees | charged on **both legs**, per section 5 |
| Bar alignment | Binance 30-minute candles close on the UTC :00/:30 grid and the live `CandleBuilder` floors samples to that same grid, so a sampled live bar and its historical counterpart describe the same window |

### 4.1 What the backtest says

120 days of real Binance 30-minute data, 8 majors, committed in `data/`. These are
the numbers a judge can reproduce from this repository:

| strategy | in-sample | out-of-sample | in-sample drawdown | round trips (IS/OOS) |
|---|---|---|---|---|
| reversion, engine defaults | -17.50% | -12.61% | 17.50% | 421 / 178 |
| reversion + Rule 4 gate | -13.45% | -11.86% | 5.60% | 333 / 144 |
| reversion + gate + no shorts | -5.13% | -5.27% | 5.60% | 163 / 55 |
| **momentum + our `.env` (shipped)** | **-2.20%** | **-0.55%** | **2.50%** | 37 / 29 |

Each row is a single change on top of the one above it, so the table also serves
as the ablation record. Fees paid in-sample fall from 11,273 to 1,102 across those
rows purely because the strategy trades less.

### 4.2 The cost measurement that decided the design

`scripts/edge_analysis.py` measures the **edge itself** rather than the strategy's
P&L, by walking forward from every entry without the stop:

```
570 Rule 2 long entries, ADX < 25
  Z reaches the -0.25 exit target within 24 bars   64.9%
  median bars to revert                            9   (~4.5 hours)
  mean gross completed move                        -0.036%
  round trip at market prices                       0.30%
```

**The mean completed move is smaller than the round-trip cost**, and it is in fact
already slightly negative before costs. Splitting the timeline into five
consecutive folds and four entry thresholds gives 20 cells, of which **19 are
negative**. Loosening the entry threshold makes it worse monotonically; tightening
it to `z >= 3` finally produces a positive gross move of +0.030%, which is still an
order of magnitude below the 0.30% it has to clear.

That is why section 5 and section 2 both matter more than the entry threshold: the
binding constraint is cost, not signal strength.

## 5. Transaction costs: maker and taker

The competition's fee schedule is **0.1% taker** and **0.05% maker**. We model
costs explicitly rather than leaving them to the broker:

```
round trip, market orders   = 2 x (0.1% taker + 0.05% slippage)              = 0.30%
round trip, maker entry     = 0.1% taker + 0.05% maker + 2 x 0.05% slippage  = 0.25%
```

| Term | Value used | Where it is set |
|---|---|---|
| `TAKER_FEE` | 0.1% per side | `config.py`, applied in `backtest._fill` as `notional x taker_fee` |
| `MAKER_FEE` | 0.05% per side | `config.fee_for(order_type)` selects taker or maker by order type |
| `SLIPPAGE_BPS` | 5 bps per side | applied as an adverse price move on every fill |
| `--spread-bps` | 5 bps assumed | a synthetic half-spread around each close, so the backtest quotes a bid and an ask |

Four consequences we want on the record:

1. **Fees are charged on both legs**, entry and exit, on the filled notional - not
   once per round trip. The cost of a trade is stated gross and net everywhere.
2. **The cost is charged inside the entry gate, not just in the accounting.** Rule
   4's deviation gate exists so that a trade is only taken when the expected move
   clears the round trip. The scored strategy carries this further: it computes
   `coverage = gain / cost` at the fill price and refuses the entry unless coverage
   clears a multiple of the cost.
3. **Passive entries are supported and measured.** `LIMIT_ENTRIES=1` posts the
   entry as a resting bid (`LIMIT_ENTRY_OFFSET_BPS`, `LIMIT_ENTRY_TIMEOUT_BARS`)
   and pays 0.05% instead of 0.1%. Exits always cross - a stop that rests is not a
   stop. Measured on the same signals, a maker entry is worth about **+0.047% per
   trade**, which is real and is roughly one seventh of the loss, not a fix.
4. **The fill model is stated conservatively.** A limit fill is reported under two
   readings: `touch` assumes we are first in the queue, `through` demands the bar
   trade strictly through our price. We quote the `through` number when deciding,
   because "the low touched my bid" does not mean we were filled.

Because the measured edge is smaller than the fee, **every design change was
evaluated on net-of-cost terms**. That is also why the current parameters trade
less often than the original ones: the cheapest trade is the one not taken.

## 6. Risk management

Rules 4-12 ship as config-driven defaults in `roostoo/risk.py`, because the engine
cannot run without a risk layer, a sizing policy and some exit. They are written to
be replaced wholesale by whoever owns them: nothing in `engine.py` or `backtest.py`
assumes the current policy.

| Rule | Implementation | State |
|---|---|---|
| 1 top-8 by 24h turnover, spread <= 0.1%, depth | `universe.py` | implemented; depth needs a provider |
| 2 SMA48/Std48 z-score entry, mean exit at -0.25/+0.25 | `strategies/mean_reversion.py` | implemented, direction as section 2.2 |
| 3 ADX(14) < 25 trend filter | `indicators.adx` | implemented |
| 4 `abs(Close-SMA48)/Close > 0.6%` | `indicators.price_deviation_pct` | implemented, default on |
| 5 stop at 1.5 x ATR(14) | `risk.protective_exits` + `PositionSizer` | `STOP_ATR_MULT` (we run 2.5 - see below) |
| 6 time stop at 12 bars | `risk.protective_exits` | `MAX_HOLD_BARS` (we disable it - see below) |
| 7 max loss 0.5% of NAV per trade | `PositionSizer` | risk-first sizing |
| 8 <= 15% NAV per coin | `RiskManager.evaluate` | `MAX_PAIR_WEIGHT` |
| 9 <= 60% NAV gross | `RiskManager.evaluate` | `MAX_GROSS_EXPOSURE` |
| 10 <= 4 positions | `RiskManager.evaluate` | `MAX_OPEN_POSITIONS` |
| 11 -2% day halts entries | `RiskManager.observe` | `MAX_DAILY_LOSS_PCT`, UTC+8 day boundary |
| 12 no re-entry for 2 bars | `RiskManager.record_exit` | `COOLDOWN_BARS` |
| supp. ReturnShock / VWAPGap / RelVolume / VolExpansion | `indicators.py` | maths implemented and tested; all off by default |

**Sizing.** The position is sized so that the distance to the actual stop, times
the quantity, equals 0.5% of NAV - so the stop distance is an input to the size,
not an afterthought. When the risk budget binds before the caps do, the caps win
instead, and the binding constraint is reported in the journal. If a stop cannot be
computed at all, the entry is refused (`REFUSE_ENTRY_WITHOUT_STOP`), because the
alternative is a full-size position with no stop, which is the opposite of
risk-first.

**Order of protective exits.** The kill switch, then the ATR stop, then the
trailing stop, then the time stop, all evaluated before any new entry in the same
cycle - so freeing a slot always beats wanting one.

**Two deliberate deviations from the playbook, both documented.** We widen the ATR
stop to 2.5x because the measured 1.5x stop fires inside the noise at an average of
4.2 bars with a 0% win rate, and we disable the 12-bar time stop because the mean
reversion completes in a median of 9-10 bars. Both are configuration values, both
are reversible with one line, and both are flagged for the team member who owns
Rule 5-6. Anyone who wants the playbook values exactly should set
`STOP_ATR_MULT=1.5` and `MAX_HOLD_BARS=12`.

**Kill switches.** A permanent halt on drawdown or an invalid state stops new
business but keeps managing what is held; Rule 11's daily halt is permanent for the
trading day. Both are persisted, so a restart cannot silently clear a halt, and the
exit status tells the supervisor the difference between "deliberately halted" and
"crashed, please restart".

## 7. The data problem, and how it is solved

The Roostoo public API exposes exactly one market-data endpoint, `/v3/ticker`,
returning a **snapshot**: `LastPrice`, `MaxBid`, `MinAsk`, 24h `Change`, 24h
`CoinTradeValue`, 24h `UnitTradeValue`. That is not enough for the playbook:

| The rules need | Roostoo provides | Resolution |
|---|---|---|
| 30-minute OHLCV candles (Rules 2-6) | nothing | `CandleBuilder` samples the ticker into 30-min bars live; `scripts/fetch_history.py` pulls real history from Binance for backtests |
| Per-bar volume (supp. Rules) | rolling 24h turnover only | live: the change in `UnitTradeValue` between samples; backtest: real Binance volume |
| Order-book depth within 0.5% (Rule 1) | no endpoint | pluggable `DepthProvider`; Binance L2 is a legitimate proxy since the venue tracks Binance |
| ADX / ATR / VWAP / VolExpansion | nothing | computed from the bars above, pure Python, no numpy |

**The organisers confirmed the mock venue's prices follow Binance.** That single
fact is what makes this design valid, and it is measured rather than assumed:
`roostoo/basis.py` compares every quoted pair against Binance on each bar, journals
the readings, and excludes any pair whose basis exceeds `BASIS_MAX_PCT` (default
1%). A wide basis is not a related-but-different market - it is the wrong symbol,
the wrong pair or a stale feed, and fading a z-score computed against it is trading
noise.

Two honest caveats. First, Rule 1's depth clause is only testable against Binance's
book, which measures *market* liquidity, not the mock venue's own depth; it answers
"is this asset liquid", not "can Roostoo absorb my order". `DEPTH_PROVIDER=none`
ships by default because the check **fails closed** - if the depth endpoint is
unreachable every pair is excluded and the bot silently stops trading, which is a
bad failure mode inside a scored window. Second, a live candle is built from
periodic ticker samples, so intrabar highs and lows can be missed; volume-derived
indicators are the most sensitive to that and we treat them as unvalidated.

## 8. Operations

```bash
python run_live.py --check             # read-only: proves signing, clock and universe, sends no orders
python run_live.py --no-seed           # cold start, build bars from live samples
python run_live.py --flatten-on-exit   # deliberate stop: close the book
```

The competition requires **at least 8 active trading days**, so the bot runs under
systemd with `Restart=always` and is deployed per `deploy/AWS_DEPLOY.md`. The unit
deliberately does **not** pass `--flatten-on-exit`: a crash-loop must never
liquidate the book. The host clock must be within 60 seconds of the exchange or
every signed request is rejected; `--check` verifies this and `chrony` is set up in
the deploy guide.

Pair names must use the venue's own quoting, which is **`/USD`**, not `/USDT` -
`ROOSTOO_PAIRS=BTC/USD,ETH/USD,...`. If none of the configured pairs are tradable
the engine says so and falls back to turnover-based discovery, which is a
deliberate fallback but a silent change of universe, so the warning is worth
watching.

## 9. Reproducing every number in this document

```bash
# 1) no keys needed: exercise the whole chain against the offline simulator
python run_live.py --mock --cycles 20

# 2) real history from Binance (no API key required)
python scripts/fetch_history.py --days 365 --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT

# 3) the ablation table in section 4.1 (first 75% in-sample, last 25% out-of-sample)
python run_backtest.py --data-dir data --oos-frac 0.25

# 4) the per-exit-reason forensics
python scripts/analyze_backtest.py reports/trades_out-of-sample.csv

# 5) the cost measurement in section 4.2
python scripts/edge_analysis.py --data-dir data
python scripts/edge_analysis.py --data-dir data --maker      # the passive-entry comparison

# 6) parameter sweeps; on Windows pass the grid as @file because PowerShell strips
#    the quotes from inline JSON (reports/grid_stop.json is a worked example)
python run_sweep.py --config-grid "@reports/grid_stop.json"

# 7) with keys: read-only verification before any order is sent
copy .env.example .env      # fill in ROOSTOO_API_KEY and ROOSTOO_SECRET_KEY
python run_live.py --check
```

## 10. Limitations and honest disclosure

* **The strategy loses money.** Every configuration measured is negative on both
  halves of the data. Section 4.2 explains why: the mean completed move is smaller
  than the round-trip cost, and 19 of 20 fold-by-threshold cells are negative.
* **The backtest dataset in this repository is 120 days.** A 2-year dataset was used
  for the direction decision in section 2.2 but is not committed (market data is
  git-ignored). Absolute returns are not comparable between the two; only the
  ordering of the compared policies is stable.
* **Live execution has not been validated end to end.** Signing, clock sync and
  universe selection are verified; real fills, the live flatten path and a
  multi-day unattended run are not. Do not treat this as production-ready.
* **Known open engineering issues** are listed in the pre-delivery review: a failed
  flatten currently ends the process rather than continuing to manage the position,
  and mock and live share a state directory by default.
* **A statistic we explicitly do not rely on.** It is true that every trade which
  reached the mean-reversion target was profitable, but that is a post-hoc
  selection on the exit outcome and cannot demonstrate that the entry factor works.
  The cost measurement in section 4.2 is the test we do rely on.
* **Not investment advice, and not a live-money system.** This targets Roostoo's
  mock exchange with a virtual portfolio. Do not point it at real capital without
  independent validation.

## 11. Layout

```
run_live.py                 live loop entry point (--check does read-only verification)
run_backtest.py             in-sample / out-of-sample backtest + report
run_sweep.py                parameter sweep & ablation, holdout-aware
scripts/fetch_history.py    Binance 30m klines -> CSV (pagination, retries, synthetic fallback)
scripts/analyze_backtest.py round-trip forensics: exit-reason mix, fee drag, holding time
scripts/edge_analysis.py    measures the edge itself against the round-trip cost
scripts/scan_secrets.py     content and history secret scanner (used by CI)
deploy/                     systemd unit + AWS EC2 guide
roostoo/
  client.py       signed REST (HMAC-SHA256), retries, throttle, server-time sync
  simulator.py    in-process mock exchange: same surface, same fees, offline
  candles.py      OHLCV bars; live bar building and CSV loading
  basis.py        cross-venue basis monitor (the venue is confirmed to track Binance)
  indicators.py   SMA/EMA/RSI/z-score + Wilder ADX, ATR, VWAP, VolExpansion, ReturnShock
  metrics.py      Sharpe / Sortino / Calmar / drawdown + the competition composite
  universe.py     Rule 1: turnover ranking, spread ceiling, depth provider
  risk.py         NAV, position book, sizing, caps, protective exits, kill switch
  strategies/     strategy interface + the z-score strategy and its scored variant
  engine.py       the autonomous decision loop
  journal.py      append-only audit trail (JSONL + trades.csv)
  backtest.py     event-driven backtester sharing the live objects
tests/            431 unittest cases, stdlib only
docs/SECURITY.md  how credentials are handled, and what to do if one leaks
```

**Never commit a key.** `.env` is git-ignored and is the only place credentials
live; `.env.example` is committed with empty values. Three guards enforce that - a
pre-commit hook, `scripts/publish.ps1` and a `secret-scan` CI job - all described in
[docs/SECURITY.md](docs/SECURITY.md).

## 12. Tests and continuous integration

```bash
python -m unittest discover -s tests -t .     # 431 tests, no network, no sleeping
```

Includes the HMAC signature reproduced byte-for-byte from Roostoo's published test
vector - the failure mode that would otherwise cost a day of the competition to
diagnose on live keys - and `tests/test_state_safety.py`, which pins the guarantees
that a bad startup or a malformed balance response cannot destroy the stored book.

CI runs on every push and pull request: the unit suite on Python 3.10 and 3.13, a
compile pass, a check that all text files are BOM-free UTF-8 with no re-encoding
artefacts, a configuration test that a missing key is rejected, and two secret
scanners (one over the working tree, one over git history). `main` is protected by
a ruleset requiring a pull request, one approval and a green `ci` check.

## 13. License

MIT - see [LICENSE](LICENSE). The competition requires the submitted repository to
be open source; a public repository without a licence is legally "all rights
reserved", so this file is what makes that claim true. Swap it for Apache-2.0 or
GPL if the team prefers, but do not remove it.
