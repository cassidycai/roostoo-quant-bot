# Deploying on the AWS EC2 instance Roostoo provides

The rulebook: *"Each team will be provided with an AWS sub-account to launch an
EC2 instance for hosting your bot... You are required to deploy your bot on an
AWS VM and ensure it executes trades automatically."*

## What the Oct 1 email is likely to contain

The organisers said teams get an **AWS sub-account** and that the competition runs
"on AWS cloud infrastructure (provisioned by Roostoo)". The Luma page links an
official *"Hackathon Guide: How to Sign In AWS and Launch Your Bot"*, so expect:

1. AWS sub-account credentials (account id / IAM user / console sign-in URL).
2. EC2 guidance — region, instance type, and whether they bill your sub-account.
3. **The Roostoo `API_KEY` / `SECRET_KEY`.** Without these nothing can trade; the
   docs say keys are issued by request to `jolly@roostoo.com`.
4. Possibly the `mock-api.roostoo.com` allow-list requirements.

**Ask in the WhatsApp group during the Sep 18 workshop** (or the Oct 1 prep
window) about three things this repo cannot answer:

* whether an **order-book depth endpoint** exists (the public docs have none —
  the `available_sub` socket.io block is commented out), because Rule 1's
  "depth within ±0.5% > $X" depends on it.

**Already answered:** **short positions are permitted** — confirmed with the
organisers. The v6 endpoint error string `this competition does not allow short
positions` is stale documentation, so `allow_short` stays at its default of true
and the short leg is expected to run. `python3 run_live.py --check` verifies the
short endpoints respond before the competition window.

**Already answered:** the mock venue's prices **track Binance**. That is why
Binance klines are valid research data here and why Binance's L2 book can stand in
for the missing depth endpoint — with the caveat that it measures market
liquidity, not the venue's own book. `python3 run_live.py --check` verifies the
basis per pair on the day; a reading beyond ~0.1% means the symbol mapping or feed
freshness needs investigating before the bot trades on it.

## 1. Launch

Ubuntu 22.04/24.04 LTS, `t3.small` is ample (the bot is a 60-second loop).
Open **no inbound ports**: the bot is outbound-only. Keep the security group
default-deny.

```bash
sudo apt-get update && sudo apt-get install -y python3 chrony
sudo timedatectl set-timezone Asia/Hong_Kong
```

The timezone is only there so your own `date`/`journalctl` output reads in the
trading day. It does **not** affect Rule 11: the day boundary comes from
`TRADING_DAY_OFFSET_HOURS` (a fixed UTC+8 offset applied to the millisecond
timestamp), so the bot computes the same boundary whatever the host clock is set
to.

### Clock accuracy is not optional

Every signed request carries a millisecond timestamp and the exchange rejects
anything more than **60 seconds** away from its own clock:

```javascript
if (abs(serverTime - timestamp) <= 60*1000) { /* process */ } else { /* reject */ }
```

```bash
sudo systemctl enable --now chrony
chronyc tracking | head -4
python3 run_live.py --check      # prints the measured offset
```

The client also measures the offset at startup and applies it, so a small skew is
survivable — a large one is not.

## 2. Install

```bash
sudo mkdir -p /opt/roostoo-quant-bot && sudo chown "$USER" /opt/roostoo-quant-bot
git clone <your-fork-url> /opt/roostoo-quant-bot
cd /opt/roostoo-quant-bot

cp .env.example .env && chmod 600 .env
nano .env        # ROOSTOO_API_KEY, ROOSTOO_SECRET_KEY, ROOSTOO_PAIRS, DEPTH_PROVIDER
```

Credentials live **only** in `.env`, which is git-ignored. Never commit them, and
never paste them into a notebook or an LLM prompt.

Seed the indicators so trading can start immediately instead of after 48 bars:

```bash
python3 scripts/fetch_history.py --days 30 --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT
```

## 3. Verify before arming

```bash
python3 run_live.py --check          # read-only: clock, exchangeInfo, ticker, balance, shorts
python3 run_live.py --mock --cycles 20   # proves the loop end-to-end
python3 run_live.py --cycles 3           # three real cycles against the venue
```

`--check` sends **no orders**, which matters: the rules forbid manual API calls
that trade, and any doubt about whether a bot "called the API by hand" is a
disqualification risk under *Commit History Transparency*.

## 4. Run it supervised

```bash
sudo cp deploy/roostoo-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now roostoo-bot
systemctl status roostoo-bot
journalctl -u roostoo-bot -f
```

The unit restarts on failure (`Restart=always`) and keeps the loop alive for 14
days. It does **not** pass `--flatten-on-exit` on purpose: with an auto-restart, a
crash-loop would liquidate the book on every restart.

## 5. Operating during the competition

```bash
tail -f logs/bot.log                         # human-readable
ls journal/                                  # decisions-YYYYMMDD.jsonl, trades-YYYYMMDD.csv
python3 -c "from roostoo.journal import read_events; print(len(read_events('journal/decisions-YYYYMMDD.jsonl')))"
```

* **Rule compliance** wants at least 8 active trading days with trades each day.
  Watch the `cycle`/`trade` counts in the journal rather than the leaderboard; a
  stalled bot looks identical to a flat one from the outside.
* **To iterate a strategy**, commit the change, redeploy, then restart. The
  position book and risk state are restored from `journal/<mode>/` (see below), so
  a restart does not lose stops or cooldowns.
* **Emergency stop that keeps positions:** `sudo systemctl stop roostoo-bot`.
* **The kill switch halting is normal, not a crash.** When `MAX_DRAWDOWN_PCT` is
  breached the bot cancels its entry orders, flattens, persists `halted: true` and
  exits with status `3`, which the unit declares a clean stop
  (`SuccessExitStatus=3`) so `Restart=always` does not restart it. It keeps running
  until the account is genuinely clean -- no position, no venue order -- so a
  halted bot with an order it could not cancel stays up and logs
  `halted but not finished`. `systemctl status` will then say `active`, not
  `inactive (dead)`, and the journal holds the halt reason. Restarting cannot help
  -- clear it by setting `"halted": false` in
  **`journal/live/engine_state.json`** (the simulator uses `journal/mock/`), and
  only after understanding why the drawdown happened. A genuine crash exits with
  any other status and is still restarted.
* **State lives in a per-mode subdirectory.** `journal/live/` and `journal/mock/`
  each hold their own `positions.json` and `engine_state.json`, so a simulator run
  cannot overwrite the live book. Upgrading from a version that wrote
  `journal/positions.json` migrates those files into the right subdirectory on the
  first live start and leaves a `*.pre-migration` backup beside the original. A
  mock run deliberately leaves unstamped legacy state alone, so smoke-testing
  before a real start cannot take the live book with it.
* **Re-fetch history before a restart you expect to be long.** The seed is refused
  when the CSV's last bar is more than 48 bars (24h) old, when the last bar has not
  closed yet, or when the series contains a hole -- warming a closed-bar strategy
  with disjoint history makes the first signals read a distribution that no longer
  exists. The bot then starts cold and needs 24h of uptime before Rules 2-3 can
  fire, so run `python3 fetch_history.py` first. The journal records
  `seed_rejected` with the reason, and the log says which pair it was.
* **Emergency stop that closes the book:** stop the service FIRST, then flatten.
  Running it while the unit is up leaves two writers on the same account and the
  same `journal/*.tmp` files. Also note that `--cycles 1` still runs one full
  decision cycle, so it can open a new position and flatten it immediately,
  paying both spreads -- to just close the book, use the venue's own UI.
  ```bash
  sudo systemctl stop roostoo-bot
  cd /opt/roostoo-quant-bot && python3 run_live.py --flatten-on-exit --cycles 1
  ```

## 6. Cost control

The instance runs for the full window; stop it when the competition ends
(`sudo shutdown -h now`) unless the team continues on the sponsored
infrastructure. Set a CloudWatch billing alarm on the sub-account's budget.
