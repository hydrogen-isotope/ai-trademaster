# Security and licensing rules

This repository holds trading software. Its biggest risks are leaked credentials, accidental live orders, and redistributing licensed market data. These rules are enforced in code wherever possible.

## 1. What may and may not be in git
| May be committed | Never committed (blocked by `.gitignore` + `tools/precommit_scan.py` + CI) |
|---|---|
| Source code, tests, docs, synthetic-data configs | `data/`: real market data (Yahoo/NASDAQ bars, quotes, forward-test logs with prices) |
| `config/default.json`, exchange skeletons | `models/`: trained weights, normalization stats, research results derived from real data |
| | `reports/`: run outputs; `*.jsonl`, `*.tmrp`, `*.onnx`, `*.parquet`, `*.dbn` |
| | Any credential: `.env*`, `*.pem/.key/.pfx/.p12`, `secrets/`, `ibkr*.json`, `config/live*.json`, `config/*.local.json` |

**Why data stays out:**
- Yahoo's terms allow personal, non-commercial use without redistribution.
- Exchange data (Nasdaq, SGX) and broker data (IBKR) are licensed per subscriber, and redistribution or public display needs separate licences.
- Models trained on that data are kept private too.

This is general guidance, not legal advice: read each vendor's agreement before sharing anything derived from it.

**Repository visibility: public.** Everything licensed, derived or personal lives under **`private/`**, which is git-ignored as a whole and blocked by the scanner and CI:
- `private/data/`: market data, forward-test logs;
- `private/models/`: trained weights, normalization stats, research JSON;
- `private/reports/`: run outputs.

A fresh clone has none of it: regenerate synthetic data with `trademaster generate`, and fetch or train your own real-data artifacts under your own data licence. Before every push, `git ls-files` must show nothing from `private/`. History is forever.

## 2. Secrets
- No credentials in code, config or command lines. Use environment variables or the Windows Credential Manager, read at runtime.
- **IBKR:** the app never sees your IBKR password. Login and 2FA happen in TWS / IB Gateway; the app talks to the local API socket.
- Rotate anything that is ever committed by mistake. Deleting it in a later commit does not remove it from history.

## 3. Pre-commit and CI checks
- `git config core.hooksPath .githooks` (already set in this clone) runs `tools/precommit_scan.py` on every commit. It blocks data/model paths, files over 2 MB, and patterns for private keys, GitHub/AWS/Google/Slack/API tokens, hard-coded passwords, and IBKR-style account numbers.
- **GitHub Actions (`.github/workflows/ci.yml`):** the same scan over all tracked files, then build and test. Least-privilege token (`contents: read`), pinned action versions.
- **Recommended GitHub settings:** 2FA on the account; branch protection on `main` (require CI, no force-push); Dependabot alerts; secret scanning with push protection and CodeQL where your plan offers them.

## 4. Live-trading safeguards (for the IBKR adapter — not built yet)
Already in the engine:
- the simulated venue is the only adapter;
- `trading.live_enabled=false`, and the engine refuses any non-simulated adapter;
- independent risk limits (position, gross, daily loss, order rate, price collar, tick/lot checks);
- the kill switch;
- stale-data and missed-deadline order suppression;
- an append-only decision audit log.

Required before any IBKR connection:
1. **Paper account first.** The adapter accepts only the paper API ports (IB Gateway 4002 / TWS 7497) unless live mode is explicitly enabled.
2. **Read-only API** enabled in IB Gateway until order sending is approved; **Trusted IPs = 127.0.0.1 only**; API socket never exposed to the network.
3. **Live mode needs three things:**
   - `trading.live_enabled=true` in a local, uncommitted config;
   - a `trading.live_confirm` phrase that names the account type and date;
   - an interactive confirmation at start-up.

   No unattended switch to live.
4. **Hard caps compiled into the adapter,** independent of config: max order notional, max orders per minute, symbol whitelist, long-only flag, max daily loss. Any breach → cancel all and stop.
5. **Reconcile** positions and open orders with IBKR at start, after every reconnect and periodically. On mismatch: stop trading and alert.
6. **Tamper-evident audit:** hash-chain each decision record; keep IBKR execution reports alongside our decisions.
7. Small-size, supervised sessions first; the decision to go live is yours alone.

## 5. Local services
- The dashboard binds to `127.0.0.1` only and has no order entry. Do not port-forward it. For remote viewing use the remote-control tooling, not an open port.
- GPU work is gated off until the display driver has been stable for 7 days (`docs/GPU-STABILITY.md`).
