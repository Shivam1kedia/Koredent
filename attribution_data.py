"""
attribution_data.py — Sprint 17, step 1: build the evidence. READ-ONLY on the repo.

Reads git history and downloads prices. Writes ONLY into attribution_out/
(do not commit that folder). Produces no results — every number that answers
a Sprint 17 question is computed later, by scripts that run only after the
predictions are pre-registered.

  attribution_out/panel.csv.gz     one row per (snapshot date, ticker): what
                                   Kordent knew and said on that day
  attribution_out/prices.csv.gz    wide daily ADJUSTED closes from 2021-01-01
  attribution_out/iima_daily.csv   IIM Ahmedabad four factors (validation only)
  attribution_out/data_log.txt     what was built, what failed

Run from the repo folder (GitHub Desktop -> Repository -> Open in Command Prompt):
    py attribution_data.py
Re-runnable: the price download resumes where it stopped, and tops up new days.

DECISIONS FIXED HERE (Sprint 17 discussion, 2026-09-21)
  * Snapshot = the LAST commit of universe_scored.csv on each UTC calendar
    date — the same rule as backtest_runner.load_cohorts.
  * Scores are taken AS STORED that day, never re-scored. The question is
    "did what Kordent actually said work", and re-scoring old rows with
    today's code is a different engine (reconcile.py declares the v4 break).
  * `investable` = selector._tier1 with affordability switched off. The
    affordability cut (one share <= one SIP instalment) is a property of a
    user, not of the engine being tested.
  * Price universe = every ticker that ever had market cap >= Rs 100 Cr in
    any snapshot, plus the benchmark ETFs and indices. The factors are built
    on the BROAD market, not on Kordent's investable list: the investable list
    removes loss-makers and red-flag companies, which is where much of the
    cheapness premium lives, and a scoreboard built without them would
    understate the tilt we are trying to catch.
  * History from 2021-01-01: five years of daily data covers a 60-month beta
    and a 12-month momentum lookback before the first snapshot.
"""
import os
import subprocess
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import yfinance as yf

import selector

# yfinance prints one "possibly delisted" line per missing ticker. A few
# thousand of those bury the progress lines; failures are counted and written
# to price_failures.txt instead.
import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

OUT = "attribution_out"
ARCHIVE = "universe_scored.csv"
PANEL_FILE = os.path.join(OUT, "panel.csv.gz")
PRICE_FILE = os.path.join(OUT, "prices.csv.gz")
IIMA_FILE = os.path.join(OUT, "iima_daily.csv")
LOG_FILE = os.path.join(OUT, "data_log.txt")

PRICE_START = date(2021, 1, 1)
RETRY_START = date(2026, 6, 1)      # short window: covers every snapshot
PRICE_MCAP_FLOOR = 100e7            # Rs 100 Cr — the broad-market floor
FETCH_CHUNK = 40
SAVE_EVERY_CHUNKS = 10
# A batch where EVERY ticker fails is the signature of Yahoo blocking this
# machine, not of forty delisted stocks. Back off, and stop after a run of
# them — pushing on through a block only lengthens it.
BLOCK_PAUSE_S = 60
BLOCK_STOP_AFTER = 4

BENCHMARK_TICKERS = [
    "^NSEI",            # Nifty 50 index (price)
    "^CRSLDX",          # Nifty 500 index (price)
    "NIFTYBEES.NS",     # Nifty 50 ETF — backtest_runner's benchmark
    "MID150BEES.NS",    # Midcap 150 ETF — selector.BENCHMARKS
    "SMALLCAP.NS",      # Smallcap 250 ETF — selector.BENCHMARKS
]

IIMA_URL = ("https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/DATA/"
            "2025-12_FourFactors_and_Market_Returns_Daily_SurvivorshipBiasAdjusted.csv")

PANEL_COLS = [
    "ticker", "name", "sector", "industry", "score", "score_continuous",
    "schema_version", "market_cap", "pb", "pe", "price", "beta",
    "quality_pass", "is_stale", "data_as_of", "years_of_data",
    "quality_axis", "growth_axis", "price_axis", "safety_axis",
    "graham_pass", "greenblatt_pass", "dorsey_pass", "trajectory_pass", "lynch_pass",
    # Business fundamentals. The event study uses these to tell a BUSINESS-driven
    # score change (reported numbers moved) from a PRICE-driven one (only the
    # price moved, which moves score through the price terms by construction).
    "revenue_y0", "revenue_y1", "net_income_y0", "net_income_y1",
    "equity_y0", "total_debt_y0", "eps",
]

os.makedirs(OUT, exist_ok=True)
_log_lines = []


def log(msg=""):
    print(msg)
    _log_lines.append(str(msg))


def git(args):
    r = subprocess.run(["git"] + args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {r.stderr.strip()[:300]}")
    return r.stdout


# ══════════════════════════════════════════════════════════════════════════
# 1. PANEL — every snapshot, as it was on its day
# ══════════════════════════════════════════════════════════════════════════
def snapshot_commits():
    """{utc_date: (sha, commit_ts)} — last commit per calendar date."""
    raw = git(["log", "--format=%H|%cI", "--", ARCHIVE])
    by_date = {}
    for line in raw.strip().splitlines():            # newest first
        if "|" not in line:
            continue
        sha, iso = line.split("|", 1)
        ts = pd.Timestamp(iso.strip()).tz_convert("UTC")
        d = ts.date()
        if d not in by_date:
            by_date[d] = (sha.strip(), ts)
    return by_date


def investable_mask(df):
    """selector._tier1 with affordability off. Returns a boolean Series aligned
    to df.index, or None if this snapshot lacks a column the floor reads (the
    oldest snapshots predate some of them)."""
    try:
        kept = selector._tier1(df.copy(), float("inf"), {})
        return df.index.isin(kept.index), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def build_panel():
    by_date = snapshot_commits()
    log(f"[PANEL] {len(by_date)} snapshot dates "
        f"({min(by_date)} -> {max(by_date)})")
    frames = []
    for d in sorted(by_date):
        sha, ts = by_date[d]
        df = pd.read_csv(StringIO(git(["show", f"{sha}:{ARCHIVE}"])), low_memory=False)
        df = df[df["ticker"].notna()].drop_duplicates("ticker", keep="last").reset_index(drop=True)

        inv, err = investable_mask(df)
        if inv is None:
            log(f"  {d}  investable=UNKNOWN ({err[:80]})")
            inv_col = pd.Series(pd.NA, index=df.index, dtype="boolean")
        else:
            inv_col = pd.Series(inv, index=df.index, dtype="boolean")

        keep = [c for c in PANEL_COLS if c in df.columns]
        out = df[keep].copy()
        for c in PANEL_COLS:
            if c not in out.columns:
                out[c] = np.nan
        out["investable"] = inv_col
        out.insert(0, "commit_utc", ts.strftime("%Y-%m-%d %H:%M"))
        out.insert(0, "snap_date", d.isoformat())
        frames.append(out)

        sv = pd.to_numeric(df.get("schema_version"), errors="coerce") \
            if "schema_version" in df.columns else pd.Series(dtype=float)
        sv = int(sv.mode().iloc[0]) if len(sv.dropna()) else None
        n_inv = int(inv_col.sum()) if inv_col.notna().any() else None
        n_pick = int((inv_col.fillna(False) & (pd.to_numeric(df["score"], errors="coerce") >= 4)).sum()) \
            if n_inv is not None else None
        log(f"  {d}  schema={sv}  rows={len(df)}  investable={n_inv}  investable&score>=4={n_pick}")

    panel = pd.concat(frames, ignore_index=True)
    panel.to_csv(PANEL_FILE, index=False)
    log(f"[PANEL] wrote {PANEL_FILE}: {len(panel):,} rows x {panel.shape[1]} cols")
    return panel


# ══════════════════════════════════════════════════════════════════════════
# 2. PRICES — broad market, adjusted closes, resumable
# ══════════════════════════════════════════════════════════════════════════
def load_prices():
    if not os.path.exists(PRICE_FILE):
        return pd.DataFrame()
    p = pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)
    p.index.name = "date"
    return p


def save_prices(p):
    p = p.sort_index()
    p.to_csv(PRICE_FILE, float_format="%.8g")


def fetch_batch(batch, start, end):
    """{ticker: Series} of adjusted closes. Empty dict on failure."""
    try:
        hist = yf.download(batch, start=start.isoformat(), end=end.isoformat(),
                           progress=False, auto_adjust=True, group_by="column",
                           threads=False)
    except Exception as e:
        log(f"    batch failed: {type(e).__name__}: {str(e)[:120]}")
        return {}
    if hist is None or hist.empty:
        return {}
    close = hist["Close"] if "Close" in hist.columns.get_level_values(0) else hist
    if isinstance(close, pd.Series):
        close = close.to_frame(batch[0])
    got = {}
    for t in close.columns:
        s = close[t].dropna()
        if len(s):
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            got[str(t)] = s[~s.index.duplicated(keep="last")]
    return got


class YahooBlocked(Exception):
    pass


def run_fetch(prices, tickers, start, end, label, chunk=FETCH_CHUNK, block_check=True):
    """block_check only makes sense on MIXED batches. A retry pass is made
    entirely of tickers that already failed, so empty batches there are the
    expected result, not a block — the 2026-09-21 Actions run stopped itself on
    exactly that misreading."""
    if not tickers:
        return prices, []
    log(f"[PRICES] {label}: {len(tickers)} tickers from {start}")
    pending = {}
    n_chunks = (len(tickers) + chunk - 1) // chunk
    failed = []
    dead_run = 0
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        got = fetch_batch(batch, start, end)
        failed += [t for t in batch if t not in got]
        pending.update(got)
        if block_check and len(batch) >= 5 and not got:
            dead_run += 1
            if dead_run >= BLOCK_STOP_AFTER:
                if pending:
                    new = pd.DataFrame(pending)
                    prices = new.combine_first(prices) if not prices.empty else new
                    save_prices(prices)
                raise YahooBlocked(
                    f"{dead_run} mixed batches in a row returned nothing. That is "
                    f"Yahoo blocking this machine. Progress is saved; wait ~30 "
                    f"minutes and run again — it resumes.")
            print(f"    whole batch empty ({dead_run}/{BLOCK_STOP_AFTER}) — "
                  f"pausing {BLOCK_PAUSE_S} s")
            time.sleep(BLOCK_PAUSE_S)
        else:
            dead_run = 0
        k = i // chunk + 1
        if k % SAVE_EVERY_CHUNKS == 0 or k == n_chunks:
            if pending:
                new = pd.DataFrame(pending)
                prices = new.combine_first(prices) if not prices.empty else new
                save_prices(prices)
                pending = {}
            print(f"    chunk {k}/{n_chunks} saved | {prices.shape[1]} tickers on disk "
                  f"| {len(failed)} failed so far")
        time.sleep(1.0)
    return prices, failed


def build_prices(panel):
    mc = pd.to_numeric(panel["market_cap"], errors="coerce")
    wanted = set(panel.loc[mc >= PRICE_MCAP_FLOOR, "ticker"].astype(str))
    wanted |= set(panel.loc[panel["investable"].fillna(False).astype(bool), "ticker"].astype(str))
    wanted |= set(BENCHMARK_TICKERS)
    wanted = sorted(wanted)
    log(f"[PRICES] universe to price: {len(wanted)} tickers "
        f"(ever >= Rs {PRICE_MCAP_FLOOR/1e7:.0f} Cr, or ever investable, plus benchmarks)")

    prices = load_prices()
    end = date.today() + timedelta(days=1)
    have = set(prices.columns)

    # Fresh tickers: full history.
    fresh = [t for t in wanted if t not in have]
    # Shuffled, reproducibly. Sorted order clumps old BSE codes (507xxx-512xxx)
    # into the same batches, and a batch of forty genuinely unlisted codes would
    # look exactly like a block. Mixed batches make "all empty" mean "blocked".
    import random
    random.Random(20260921).shuffle(fresh)
    prices, failed = run_fetch(prices, fresh, PRICE_START, end, "full history")

    # Top-up: tickers on disk whose series ends more than 3 days ago.
    if not prices.empty:
        last = prices.apply(lambda s: s.last_valid_index())
        stale = [t for t in wanted if t in prices.columns
                 and last.get(t) is not None
                 and last[t].date() < date.today() - timedelta(days=3)]
        if stale:
            start = min(last[t].date() for t in stale) - timedelta(days=7)
            prices, _ = run_fetch(prices, stale, start, end, "top-up")

    # ── Failures. Measured 2026-09-21 on Actions: ~22% of every batch fails
    # steadily (not in bursts), so these are specific tickers, not a block.
    # Two fallbacks, in order, and every series records where it came from:
    #   (1) Yahoo again, SHORT window from RETRY_START, batches of 5
    #   (2) backtest_price_cache.csv — the same Yahoo closes, fetched by the
    #       backtest from 2026-07-06; enough for in-window returns, NOT for the
    #       12-month momentum or volatility lookback.
    source = {t: "yahoo_full" for t in prices.columns}
    if failed:
        log(f"[PRICES] {len(failed)} failed the full-history fetch: "
            f"{sum(t.endswith('.BO') for t in failed)} .BO, "
            f"{sum(t.endswith('.NS') for t in failed)} .NS, "
            f"{sum(not t.endswith(('.BO', '.NS')) for t in failed)} other")
        time.sleep(30)
        prices, failed = run_fetch(prices, failed, RETRY_START, end,
                                   "retry, short window", chunk=5, block_check=False)
        for t in prices.columns:
            source.setdefault(t, "yahoo_short")

    if failed and os.path.exists("backtest_price_cache.csv"):
        c = pd.read_csv("backtest_price_cache.csv", parse_dates=["date"])
        c = c[c["ticker"].isin(failed)]
        if len(c):
            w = c.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
            w.index = pd.to_datetime(w.index).normalize()
            prices = w.combine_first(prices)
            for t in w.columns:
                source[t] = "backtest_cache"
            save_prices(prices)
        log(f"[PRICES] filled {c['ticker'].nunique() if len(c) else 0} from backtest_price_cache.csv")
        failed = [t for t in failed if t not in prices.columns]

    with open(os.path.join(OUT, "price_failures.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(failed))
    prov = pd.DataFrame({
        "ticker": list(prices.columns),
        "source": [source.get(t, "yahoo_full") for t in prices.columns],
        "first_date": [prices[t].first_valid_index() for t in prices.columns],
        "n_days": [int(prices[t].count()) for t in prices.columns],
    })
    prov.to_csv(os.path.join(OUT, "price_sources.csv"), index=False)
    log("[PRICES] sources: " + ", ".join(f"{k}={v}" for k, v in
                                         prov["source"].value_counts().items()))

    log(f"[PRICES] on disk: {prices.shape[1]} tickers x {prices.shape[0]} days "
        f"({prices.index.min().date()} -> {prices.index.max().date()})")
    log(f"[PRICES] never priced: {len(failed)} (listed in {OUT}/price_failures.txt)")
    for b in BENCHMARK_TICKERS:
        s = prices[b].dropna() if b in prices.columns else pd.Series(dtype=float)
        log(f"  benchmark {b:14s} " + (f"{len(s)} days from {s.index.min().date()}"
                                       if len(s) else "MISSING"))
    return prices


# ══════════════════════════════════════════════════════════════════════════
# 3. IIM-A FACTORS — validation only (they end 2025-12)
# ══════════════════════════════════════════════════════════════════════════
def fetch_iima():
    if not os.path.exists(IIMA_FILE):
        try:
            req = urllib.request.Request(IIMA_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r, open(IIMA_FILE, "wb") as f:
                f.write(r.read())
        except Exception as e:
            log(f"[IIMA] download failed: {type(e).__name__}: {e}")
            log(f"[IIMA] download it by hand into {IIMA_FILE} from:\n  {IIMA_URL}")
            return
    try:
        x = pd.read_csv(IIMA_FILE)
        log(f"[IIMA] {IIMA_FILE}: {x.shape[0]} rows | columns {list(x.columns)}")
        log("[IIMA] first and last rows:")
        log(x.head(2).to_string(index=False))
        log(x.tail(2).to_string(index=False))
    except Exception as e:
        log(f"[IIMA] unreadable: {e}")


# ══════════════════════════════════════════════════════════════════════════
# 4. COVERAGE — can the tests stand on this?
# ══════════════════════════════════════════════════════════════════════════
def coverage(panel, prices):
    log("\n[COVERAGE] Kordent picks (investable & score>=4) with usable price history")
    picks = panel[panel["investable"].fillna(False).astype(bool)
                  & (pd.to_numeric(panel["score"], errors="coerce") >= 4)]
    tick = sorted(set(picks["ticker"].astype(str)))
    n_any = sum(t in prices.columns for t in tick)
    n_1y = sum(t in prices.columns and prices[t].count() >= 250 for t in tick)
    log(f"  distinct picks ever: {len(tick)} | priced: {n_any} | with >= 1 year: {n_1y}")

    mc = pd.to_numeric(panel["market_cap"], errors="coerce")
    broad = sorted(set(panel.loc[mc >= PRICE_MCAP_FLOOR, "ticker"].astype(str)))
    n_b = sum(t in prices.columns for t in broad)
    log(f"  broad market (ever >= Rs 100 Cr): {len(broad)} | priced: {n_b}")


def main():
    log(f"attribution_data.py run {datetime.now():%Y-%m-%d %H:%M}")
    code = 0
    try:
        try:
            panel = build_panel()
        except RuntimeError as e:
            log(f"[HALT] {e}\nRun this from the git repo folder "
                f"(GitHub Desktop -> Repository -> Open in Command Prompt).")
            code = 1
            return
        fetch_iima()                    # cheap, and independent of Yahoo
        try:
            prices = build_prices(panel)
        except YahooBlocked as e:
            log(f"[PRICES] STOPPED: {e}")
            code = 2
            prices = load_prices()
        coverage(panel, prices)
    finally:
        # Always written — the first Actions run exited before this line and
        # left no record of what had failed.
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(_log_lines))
        print(f"\nLog written to {LOG_FILE}.")
    sys.exit(code)


if __name__ == "__main__":
    main()
