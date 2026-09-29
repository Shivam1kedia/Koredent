"""
BACKTEST RUNNER  (Sprint 13 rewrite — point-in-time, archive-driven)
====================================================================
Replaces the old yfinance-reconstruction backtest. That approach walked TODAY's
restated 4-year financials backwards to "reconstruct" past scores; Sprint 12
established it produces a progressively distorted impostor (restated financials
+ survivorship from a current index list). This runner does the honest thing
instead: it reads the point-in-time score archive we already commit every
weekday and measures FORWARD returns.

The question this answers, and the only one it answers:
    Did the stocks we scored 3+ on a given day go on to beat the benchmark over
    the following ~quarter — and did the low-scored ones not?

Method — BUY-AND-HOLD BY ENTRY SCORE (score drift within the horizon is ignored
on purpose):
  1. Each clean daily snapshot of universe_scored.csv (read from git history) is
     a cohort: {ticker -> score} exactly as known on date D. No restatement.
  2. Bucket the cohort by entry score. Measure each bucket's mean forward return
     from D to D+HORIZON using real historical prices, held untouched.
  3. Benchmark = NIFTYBEES buy-and-hold over the same window.
  4. Aggregate across matured cohorts. A cohort counts only once its horizon has
     fully elapsed.

Honesty, baked in not bolted on:
  - Until enough MATURED clean cohorts exist, the runner refuses to emit a
    return claim and reports "insufficient data — first signal ~<date>".
  - Delisted names with no forward price are dropped and COUNTED; that biases
    surviving-bucket returns UPWARD, and the output says so.
  - Reads only the point-in-time snapshot, so there is no current-list
    survivorship bias — the residual is delisting only.

Data source (A): git history of universe_scored.csv. Requires the workflow to
checkout with fetch-depth: 0 (full history), else only today's snapshot exists.

Outputs:
  backtest_results.csv  — one row per (cohort, score_bucket): forward return,
                          benchmark return, alpha, n_priced, n_missing.
  backtest_summary.json — aggregate per-bucket means + status + caveats.
"""

import os
import sys
import json
import math
import time
import random
import subprocess
from datetime import datetime, timedelta, date

import pandas as pd
import numpy as np
import yfinance as yf

import costs

# ─── Config ───
ARCHIVE_FILE = "universe_scored.csv"          # the git-versioned snapshot
MANIFEST_FILE = "archive_manifest.json"       # the clock (clean-snapshot record)
BENCHMARK = "NIFTYBEES.NS"                     # buyable ETF, consistent with §1
HORIZON_TRADING_DAYS = 63                      # ~1 quarter; configurable
PRICE_BUCKETS = [3, 4, 5]                      # priced in full (few names)
CONTROL_BUCKETS = [0, 1, 2]                    # sampled — control group
CONTROL_SAMPLE_PER_BUCKET = 100               # cap per control bucket per cohort
MIN_MATURED_COHORTS = 1                        # show a reading at 1 matured cohort; certainty is the overlap-adjusted band's job, not a hard gate's
MIN_CLEAN_SCHEMA = 1                            # snapshot schema_version floor
PRICE_CACHE_FILE = "backtest_price_cache.csv"  # immutable historical closes
FETCH_CHUNK = 40                               # tickers per yfinance batch
SAMPLE_SEED = 20260713                          # reproducible control sampling

# ─── Cost assumptions (Sprint 16) ───
# PRE-REGISTERED 2026-09-18, before any netted number was computed.
#
# The backtest prices buy-and-hold forward returns. Those returns were GROSS of
# every rupee of friction until now, which measured a strategy nobody can
# execute: a Rs 500 position costs 3.29% to round-trip, and the ladder's whole
# high-minus-low spread could be smaller than that.
#
# Base case Rs 5,000 per position (0.53% round trip). Sensitivity Rs 500
# (3.29%), because Rs 500 is what a minimum-SIP user across the 10-name ruin
# floor actually gets — the base case is the aspiration, the sensitivity is the
# product as shipped.
COST_POSITION_BASE = 5000.0
COST_POSITION_SENSITIVITY = 500.0
# The benchmark is ONE ETF position, not fifteen, so it carries one flat DP
# charge rather than fifteen. Netting it per-position would hand the strategy
# free alpha worth ~14 depository charges. Sized at base position x a typical
# 15-name portfolio.
COST_POSITION_BENCHMARK = COST_POSITION_BASE * 15

# Allow a forced research run even when the archive can't be read (e.g. a
# shallow checkout locally). Never wire this into a committing workflow.
FORCE = os.environ.get("KORDENT_BACKTEST_FORCE") == "1"


# ══════════════════════════════════════════════════════════════════════════
# 1. READ THE POINT-IN-TIME ARCHIVE FROM GIT HISTORY
# ══════════════════════════════════════════════════════════════════════════
def _git(args):
    """Run a git command, return stdout (str) or None on failure."""
    try:
        out = subprocess.run(["git"] + args, capture_output=True, text=True,
                             timeout=60)
        if out.returncode != 0:
            print(f"[GIT] {' '.join(args)} -> rc={out.returncode}: {out.stderr.strip()[:200]}")
            return None
        return out.stdout
    except Exception as e:
        print(f"[GIT] {' '.join(args)} failed: {e}")
        return None


def list_snapshot_commits():
    """Every commit that touched universe_scored.csv, newest first, as
    (sha, commit_date). Each such commit is one daily snapshot."""
    raw = _git(["log", "--format=%H|%cI", "--", ARCHIVE_FILE])
    if not raw:
        return []
    commits = []
    for line in raw.strip().splitlines():
        if "|" not in line:
            continue
        sha, iso = line.split("|", 1)
        try:
            d = datetime.fromisoformat(iso.strip()).date()
        except Exception:
            continue
        commits.append((sha.strip(), d))
    return commits


def load_snapshot(sha):
    """Read universe_scored.csv AS OF a commit into a DataFrame, or None."""
    raw = _git(["show", f"{sha}:{ARCHIVE_FILE}"])
    if not raw:
        return None
    try:
        from io import StringIO
        return pd.read_csv(StringIO(raw))
    except Exception as e:
        print(f"[ARCHIVE] parse failed for {sha[:8]}: {e}")
        return None


def load_cohorts():
    """Build one cohort per clean snapshot date: {'date', 'scores': {ticker:score}}.
    De-duplicates to the LAST commit per calendar date. Filters to snapshots whose
    own schema_version column is clean (>= MIN_CLEAN_SCHEMA) — the snapshot is
    self-describing, so we don't need the manifest to judge cleanliness."""
    commits = list_snapshot_commits()
    if not commits:
        return []
    seen_dates = set()
    cohorts = []
    for sha, d in commits:                      # newest first
        if d in seen_dates:
            continue
        df = load_snapshot(sha)
        if df is None or "ticker" not in df.columns or "score" not in df.columns:
            continue
        # Clean-scorer gate: schema_version stamped on every row by universe_updater.
        if "schema_version" in df.columns:
            try:
                sv = int(pd.to_numeric(df["schema_version"], errors="coerce").dropna().mode().iloc[0])
            except Exception:
                sv = 0
            if sv < MIN_CLEAN_SCHEMA:
                continue
        else:
            # Pre-schema snapshots ran the broken scorer (Sprint 12). Skip.
            continue
        scores = {}
        for t, s in zip(df["ticker"], df["score"]):
            if pd.notna(t) and pd.notna(s):
                scores[str(t)] = int(s)
        if scores:
            seen_dates.add(d)
            cohorts.append({"date": d, "scores": scores})
    cohorts.sort(key=lambda c: c["date"])       # oldest first
    return cohorts


def load_manifest():
    """The clock. Used only for reporting/projection, not for correctness."""
    if not os.path.exists(MANIFEST_FILE):
        return []
    try:
        with open(MANIFEST_FILE) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("snapshots", [])
    except Exception:
        return []


# ══════════════════════════════════════════════════════════════════════════
# 2. PRICES — fetch once per ticker over the full window, cache immutably
# ══════════════════════════════════════════════════════════════════════════
def _load_price_cache():
    if not os.path.exists(PRICE_CACHE_FILE):
        return {}, {}
    try:
        c = pd.read_csv(PRICE_CACHE_FILE, parse_dates=["date"])
        series = {}
        maxdate = {}
        for tkr, g in c.groupby("ticker"):
            s = g.set_index("date")["close"].sort_index()
            series[str(tkr)] = s
            maxdate[str(tkr)] = s.index.max().date()
        return series, maxdate
    except Exception as e:
        print(f"[CACHE] load failed, starting fresh: {e}")
        return {}, {}


def _save_price_cache(series):
    rows = []
    for tkr, s in series.items():
        for dt, px in s.items():
            rows.append({"ticker": tkr, "date": pd.Timestamp(dt).date(), "close": px})
    if rows:
        pd.DataFrame(rows).to_csv(PRICE_CACHE_FILE, index=False)


def _yahoo_symbol_map():
    """ticker -> the symbol Yahoo answers to, from the universe CSV.

    universe_updater decides this ONCE (_bse_yahoo_symbol) and writes it as
    `yf_symbol`. Numeric BSE codes stopped resolving at Yahoo around 2026-09-20;
    without this map every sampled BSE control name silently stops getting
    forward prices, n_missing climbs, and the control buckets drift toward NSE.
    Empty map (older CSV without the column) = today's behaviour, unchanged.
    """
    try:
        u = pd.read_csv(ARCHIVE_FILE, usecols=["ticker", "yf_symbol"])
        u = u.dropna()
        return {str(t): str(s) for t, s in zip(u["ticker"], u["yf_symbol"])}
    except Exception:
        return {}


def fetch_prices(tickers, start, end):
    """Return {ticker: close Series} over [start, end], using and extending the
    on-disk cache. Historical closes are immutable, so a ticker already cached
    through >= end is reused untouched; only missing/short ones are fetched."""
    series, maxdate = _load_price_cache()
    need = [t for t in tickers
            if t not in series or maxdate.get(t, date.min) < (end - timedelta(days=3))]
    print(f"[PRICES] {len(tickers)} needed | {len(tickers) - len(need)} cached | "
          f"{len(need)} to fetch")

    ymap = _yahoo_symbol_map()
    for i in range(0, len(need), FETCH_CHUNK):
        batch = need[i:i + FETCH_CHUNK]
        # Ask Yahoo by yf_symbol, store by ticker. The cache stays keyed on the
        # identity every cohort uses.
        back = {ymap.get(t, t): t for t in batch}
        batch = list(back.keys())
        try:
            # threads=False on purpose — curl_cffi's threaded path is the one that
            # differs on Linux (see the §1 tracker note); single-threaded is stable.
            hist = yf.download(batch, start=start.isoformat(), end=end.isoformat(),
                               progress=False, auto_adjust=True, group_by="column",
                               threads=False)
        except Exception as e:
            print(f"[PRICES] batch {i // FETCH_CHUNK + 1} failed: {e}")
            continue
        if hist is None or hist.empty:
            continue
        close = hist["Close"] if "Close" in hist else hist
        if isinstance(close, pd.Series):        # single ticker
            close = close.to_frame(batch[0])
        for t in close.columns:
            s = close[t].dropna()
            if len(s):
                s.index = pd.to_datetime(s.index)
                series[back.get(str(t), str(t))] = s
        time.sleep(0.5)                          # be gentle with the rate limiter

    _save_price_cache(series)
    return series


def price_asof(s, d, tol_days=10):
    """Close on the last trading day <= d. Returns None if the series doesn't
    REACH near d (last bar > tol_days before d) — i.e. the stock delisted or
    stopped trading, so there is genuinely no price at d. Without this a dead
    stock's last-ever price would masquerade as its exit price and silently
    readmit the very survivorship losses we drop-and-count."""
    if s is None or len(s) == 0:
        return None
    ts = pd.Timestamp(d)
    up_to = s[s.index <= ts]
    if len(up_to) == 0:
        return None
    if (ts - up_to.index[-1]).days > tol_days:
        return None
    return float(up_to.iloc[-1])


# ══════════════════════════════════════════════════════════════════════════
# 3. FORWARD RETURNS BY ENTRY SCORE BUCKET  (buy-and-hold)
# ══════════════════════════════════════════════════════════════════════════
def build_calendar(bench_series):
    """Canonical trading-day index from the benchmark. Cohort exit = the bar
    HORIZON_TRADING_DAYS after entry on THIS calendar, identical for every stock
    in the cohort."""
    return list(bench_series.index) if bench_series is not None else []


def exit_date_for(calendar, d):
    """The trading date HORIZON_TRADING_DAYS after d, or None if not enough
    calendar has elapsed yet (cohort not matured)."""
    ts = pd.Timestamp(d)
    pos = None
    for i, cd in enumerate(calendar):
        if cd >= ts:
            pos = i
            break
    if pos is None:
        return None
    tgt = pos + HORIZON_TRADING_DAYS
    if tgt >= len(calendar):
        return None                              # horizon hasn't elapsed
    return calendar[tgt].date()


def bucket_tickers(scores, rng):
    """{bucket: [tickers]} — 3/4/5 in full, 0/1/2 sampled to the control cap."""
    by = {b: [] for b in PRICE_BUCKETS + CONTROL_BUCKETS}
    for t, s in scores.items():
        if s in by:
            by[s].append(t)
    for b in CONTROL_BUCKETS:
        if len(by[b]) > CONTROL_SAMPLE_PER_BUCKET:
            by[b] = rng.sample(sorted(by[b]), CONTROL_SAMPLE_PER_BUCKET)
    return by


def cohort_returns(cohort, calendar, prices):
    """Per-bucket forward return for one matured cohort. Returns a list of row
    dicts, or [] if the cohort has not matured."""
    d = cohort["date"]
    xdate = exit_date_for(calendar, d)
    if xdate is None:
        return []                                 # not matured

    bench = prices.get(BENCHMARK)
    b_in, b_out = price_asof(bench, d), price_asof(bench, xdate)
    if not b_in or not b_out or b_in <= 0:
        return []                                 # can't benchmark this cohort
    bench_ret = b_out / b_in - 1.0
    # Netted at ONE ETF position, not per-stock. See COST_POSITION_BENCHMARK.
    bench_ret_net = costs.net_return(COST_POSITION_BENCHMARK, bench_ret)

    rng = random.Random(f"{SAMPLE_SEED}-{d.isoformat()}")
    buckets = bucket_tickers(cohort["scores"], rng)

    rows = []
    for b, tks in buckets.items():
        rets, missing = [], 0
        for t in tks:
            s = prices.get(t)
            p_in, p_out = price_asof(s, d), price_asof(s, xdate)
            if p_in and p_out and p_in > 0:
                rets.append(p_out / p_in - 1.0)
            else:
                missing += 1                       # delisted / no price — counted
        n = len(rets)
        # Cost is applied PER STOCK before averaging, not to the bucket mean.
        # The two differ because net_return is not linear in the gross return
        # (the sell leg is charged on the exit value), and averaging first would
        # quietly price every holding at the bucket's average outcome.
        net_base = [costs.net_return(COST_POSITION_BASE, r) for r in rets]
        net_sens = [costs.net_return(COST_POSITION_SENSITIVITY, r) for r in rets]
        rows.append({
            "cohort_date": d.isoformat(),
            "exit_date": xdate.isoformat(),
            "score_bucket": b,
            "n_priced": n,
            "n_missing": missing,
            # GROSS — the raw measurement, untouched. Costs are an assumption
            # layered on top; deleting the gross figures would make the
            # assumption unfalsifiable.
            "fwd_return": round(float(np.mean(rets)), 4) if n else None,
            "bench_return": round(bench_ret, 4),
            "alpha": round(float(np.mean(rets)) - bench_ret, 4) if n else None,
            # NET — at the two pre-registered position sizes.
            "fwd_return_net": round(float(np.mean(net_base)), 4) if n else None,
            "fwd_return_net_small": round(float(np.mean(net_sens)), 4) if n else None,
            "bench_return_net": round(bench_ret_net, 4),
            "alpha_net": (round(float(np.mean(net_base)) - bench_ret_net, 4)
                          if n else None),
            "sampled": b in CONTROL_BUCKETS,
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════
# 4. AGGREGATE + HONEST GATING
# ══════════════════════════════════════════════════════════════════════════
def project_first_reading(cohorts):
    """Projected date the FIRST cohort matures = earliest clean snapshot + one
    horizon. 'now' once anything has matured (handled by the caller)."""
    if not cohorts:
        return "unknown (no clean snapshots yet)"
    horizon_cal = round(HORIZON_TRADING_DAYS * 7 / 5)
    return (cohorts[0]["date"] + timedelta(days=horizon_cal)).isoformat()


def _is_monotonic(ladder_means):
    """True iff mean fwd return is non-increasing from bucket 5 down to 0 across
    the buckets present (>=2 needed to mean anything)."""
    present = [ladder_means[b] for b in [5, 4, 3, 2, 1, 0] if b in ladder_means]
    return len(present) >= 2 and all(present[i] >= present[i + 1]
                                     for i in range(len(present) - 1))


def _high_minus_low(bmeans):
    """mean(buckets 4,5) - mean(buckets 0,1) for one cohort, or None if either
    side is entirely absent that day."""
    hi = [bmeans[b] for b in (4, 5) if b in bmeans]
    lo = [bmeans[b] for b in (0, 1) if b in bmeans]
    if not hi or not lo:
        return None
    return float(np.mean(hi) - np.mean(lo))


def overlap_adjusted_ci(values, span_days, horizon_cal_days):
    """95% CI on the mean of per-cohort statistics, with the sample size
    discounted for window overlap. Daily cohorts overlap ~62/63 days, so the
    honest denominator is the number of NON-overlapping horizons the cohorts
    span, not the raw cohort count. Returns
    (point, lo, hi, n_cohorts, independent_quarters); lo/hi are None until n>=2."""
    xs = [v for v in values if v is not None]
    n = len(xs)
    if n == 0:
        return None, None, None, 0, 0
    point = float(np.mean(xs))
    n_eff = max(1.0, span_days / horizon_cal_days) if horizon_cal_days else 1.0
    n_eff_int = max(1, int(n_eff))
    if n < 2:
        return round(point, 4), None, None, n, n_eff_int
    se = float(np.std(xs, ddof=1)) / math.sqrt(n_eff)
    return round(point, 4), round(point - 1.96 * se, 4), round(point + 1.96 * se, 4), n, n_eff_int


def aggregate(all_rows, cohorts, matured_cohorts):
    """The ladder (mean fwd return per entry bucket) + the headline high-minus-low
    spread with an overlap-adjusted band. Cross-sectional by design: buckets in a
    cohort share the same market window, so a good/bad quarter cancels out of the
    spread. Status flips to OK at the first matured cohort; the band, not a gate,
    carries the uncertainty."""
    status = "OK" if matured_cohorts >= MIN_MATURED_COHORTS else "INSUFFICIENT_DATA"
    horizon_cal = round(HORIZON_TRADING_DAYS * 7 / 5)
    summary = {
        "status": status,
        "matured_cohorts": matured_cohorts,
        "snapshots_clean": len(cohorts),
        "min_required": MIN_MATURED_COHORTS,
        "horizon_trading_days": HORIZON_TRADING_DAYS,
        "benchmark": BENCHMARK,          # trading-day calendar spine only, not a signal
        "generated": datetime.now().isoformat(timespec="seconds"),
        "caveats": [
            "Cross-sectional ladder: within a cohort all buckets share the same "
            "window, so a good/bad quarter lifts or sinks them together and "
            "cancels out of the high-minus-low spread.",
            "Daily cohorts overlap ~62/63 days — they sharpen the estimate but are "
            "NOT independent. The band uses effective N = calendar span / horizon, "
            "not the raw cohort count.",
            "Delisted names with no forward price are dropped and counted in "
            "n_missing; this biases surviving-bucket returns UPWARD.",
            "Point-in-time snapshot => no current-list survivorship bias; residual "
            "is delisting only.",
            f"COSTS: *_net figures are net of modelled Zerodha equity-delivery "
            f"costs at Rs {COST_POSITION_BASE:,.0f} per position "
            f"({costs.round_trip_pct(COST_POSITION_BASE)*100:.2f}% round trip); "
            f"*_net_small at Rs {COST_POSITION_SENSITIVITY:,.0f} "
            f"({costs.round_trip_pct(COST_POSITION_SENSITIVITY)*100:.2f}%), which "
            f"is what a minimum-SIP user across the 10-name ruin floor gets. The "
            f"benchmark is netted as ONE Rs {COST_POSITION_BENCHMARK:,.0f} ETF "
            f"position, carrying one flat depository charge rather than fifteen. "
            f"Position sizes pre-registered 2026-09-18 before any netted number "
            f"was computed. Slippage and market impact are NOT modelled and are "
            f"not zero, so every net figure here still flatters the strategy.",
            "Gross figures are the raw measurement and are reported unchanged "
            "alongside the net ones; costs are an assumption layered on top.",
        ],
        "cost_assumption": {
            "rates_verified": costs.RATES_VERIFIED,
            "position_base_inr": COST_POSITION_BASE,
            "position_base_round_trip_pct":
                round(costs.round_trip_pct(COST_POSITION_BASE) * 100, 3),
            "position_sensitivity_inr": COST_POSITION_SENSITIVITY,
            "position_sensitivity_round_trip_pct":
                round(costs.round_trip_pct(COST_POSITION_SENSITIVITY) * 100, 3),
            "benchmark_position_inr": COST_POSITION_BENCHMARK,
            "slippage_modelled": False,
            "pre_registered": "2026-09-18",
        },
        "ladder": {},
    }

    priced = pd.DataFrame(all_rows)
    priced = priced[priced["fwd_return"].notna()] if len(priced) else priced

    if not len(priced):
        summary["ladder_monotonic"] = None
        summary["independent_quarters"] = 0
        summary["spread_high_minus_low"] = {
            "definition": "mean(buckets 4,5) - mean(buckets 0,1) forward return",
            "point": None, "ci95": None, "n_cohorts": 0, "independent_quarters": 0}
        summary["spread_high_minus_low_net"] = {
            "definition": f"same spread, net of costs at Rs {COST_POSITION_BASE:,.0f}"
                          f" per position",
            "point": None, "ci95": None, "n_cohorts": 0, "independent_quarters": 0}
        if status == "INSUFFICIENT_DATA":
            summary["first_reading_date"] = project_first_reading(cohorts)
        return summary

    # ── aggregate ladder: mean fwd return per bucket across matured cohorts ──
    ladder_means = {}
    for b in sorted(set(PRICE_BUCKETS + CONTROL_BUCKETS)):
        sub = priced[priced["score_bucket"] == b]
        if len(sub):
            m = round(float(sub["fwd_return"].mean()), 4)
            ladder_means[b] = m
            summary["ladder"][str(b)] = {
                "cohorts": int(sub["cohort_date"].nunique()),
                "mean_fwd_return": m,
                "mean_fwd_return_net": round(float(sub["fwd_return_net"].mean()), 4),
                "mean_fwd_return_net_small":
                    round(float(sub["fwd_return_net_small"].mean()), 4),
                "total_missing": int(sub["n_missing"].sum()),
                "sampled": bool(b in CONTROL_BUCKETS),
            }
    summary["ladder_monotonic"] = _is_monotonic(ladder_means)

    # ── per-cohort high-minus-low spread → overlap-adjusted band ──
    spread_by_cohort = {}
    spread_by_cohort_net = {}
    for cd, g in priced.groupby("cohort_date"):
        bmeans = {int(r["score_bucket"]): r["fwd_return"] for _, r in g.iterrows()}
        spread_by_cohort[cd] = _high_minus_low(bmeans)
        bnet = {int(r["score_bucket"]): r["fwd_return_net"] for _, r in g.iterrows()}
        spread_by_cohort_net[cd] = _high_minus_low(bnet)
    dates = sorted(pd.to_datetime(list(spread_by_cohort.keys())))
    span_days = (dates[-1] - dates[0]).days if len(dates) > 1 else 0
    point, lo, hi, n, m_eff = overlap_adjusted_ci(
        list(spread_by_cohort.values()), span_days, horizon_cal)
    summary["independent_quarters"] = m_eff
    summary["spread_high_minus_low"] = {
        "definition": "mean(buckets 4,5) - mean(buckets 0,1) forward return",
        "point": point,
        "ci95": [lo, hi] if lo is not None else None,
        "n_cohorts": n,
        "independent_quarters": m_eff,
    }
    # The spread is a DIFFERENCE of two buckets bought and sold the same way, so
    # the proportional costs largely cancel and the flat DP charge cancels
    # exactly. It should move very little. Reporting it is how that claim stays
    # falsifiable instead of assumed.
    npoint, nlo, nhi, nn, _ = overlap_adjusted_ci(
        list(spread_by_cohort_net.values()), span_days, horizon_cal)
    summary["spread_high_minus_low_net"] = {
        "definition": f"same spread, net of costs at Rs {COST_POSITION_BASE:,.0f}"
                      f" per position",
        "point": npoint,
        "ci95": [nlo, nhi] if nlo is not None else None,
        "n_cohorts": nn,
        "independent_quarters": m_eff,
    }
    if status == "INSUFFICIENT_DATA":
        summary["first_reading_date"] = project_first_reading(cohorts)
    return summary


# ══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 64)
    print("KORDENT BACKTEST — point-in-time, buy-and-hold by entry score")
    print(f"Run: {datetime.now():%Y-%m-%d %H:%M} | horizon {HORIZON_TRADING_DAYS}td "
          f"| benchmark {BENCHMARK}")
    print("=" * 64)

    print("\n--- STEP 1: Load clean snapshots from git history ---")
    cohorts = load_cohorts()
    manifest = load_manifest()
    print(f"Clean cohorts in archive: {len(cohorts)} | manifest records: {len(manifest)}")
    if not cohorts:
        msg = ("No clean point-in-time snapshots in git history. Either the "
               "checkout is shallow (need fetch-depth: 0) or the clean-scorer "
               "archive hasn't started. Nothing to backtest.")
        print(f"[HALT] {msg}")
        json.dump({"status": "NO_ARCHIVE", "detail": msg},
                  open("backtest_summary.json", "w"), indent=2)
        if not FORCE:
            return

    print("\n--- STEP 2: Fetch prices (cached, immutable historical closes) ---")
    all_tickers = set([BENCHMARK])
    _rng = random.Random(SAMPLE_SEED)
    for c in cohorts:
        for b, tks in bucket_tickers(c["scores"],
                                     random.Random(f"{SAMPLE_SEED}-{c['date'].isoformat()}")).items():
            all_tickers.update(tks)
    start = (min(c["date"] for c in cohorts) - timedelta(days=7)) if cohorts else date.today()
    prices = fetch_prices(sorted(all_tickers), start, date.today())

    print("\n--- STEP 3: Forward returns by entry-score bucket ---")
    calendar = build_calendar(prices.get(BENCHMARK))
    if not calendar:
        print("[HALT] No benchmark price series; cannot define the horizon calendar.")
        json.dump({"status": "NO_BENCHMARK"}, open("backtest_summary.json", "w"), indent=2)
        return

    all_rows, matured = [], 0
    for c in cohorts:
        rows = cohort_returns(c, calendar, prices)
        if rows:
            matured += 1
            all_rows.extend(rows)
            picks = next((r for r in rows if r["score_bucket"] == 3), None)
            print(f"  {c['date']} matured -> 3/5 fwd "
                  f"{picks['fwd_return'] if picks else '—'} vs bench {rows[0]['bench_return']}")
        else:
            print(f"  {c['date']} not matured yet (horizon not elapsed)")

    print(f"\nMatured clean cohorts: {matured} / {MIN_MATURED_COHORTS} required")

    print("\n--- STEP 4: Aggregate + honest gating ---")
    summary = aggregate(all_rows, cohorts, matured)

    pd.DataFrame(all_rows).to_csv("backtest_results.csv", index=False)
    json.dump(summary, open("backtest_summary.json", "w"), indent=2)
    print(f"Wrote backtest_results.csv ({len(all_rows)} bucket-cohort rows) "
          f"and backtest_summary.json")

    if summary["status"] == "INSUFFICIENT_DATA":
        print(f"\n[STATUS] INSUFFICIENT DATA — {matured} matured cohorts. No reading yet.")
        print(f"[STATUS] First reading projected ~{summary.get('first_reading_date','?')}.")
    elif summary["status"] == "OK":
        print(f"\n[RESULT] ladder — mean forward return by entry score "
              f"({matured} cohorts, ~{summary['independent_quarters']} independent quarters):")
        for b in ["5", "4", "3", "2", "1", "0"]:
            v = summary["ladder"].get(b)
            if v:
                tag = " (sampled)" if v["sampled"] else ""
                print(f"  {b}/5: gross {v['mean_fwd_return']:+.2%} | "
                      f"net {v['mean_fwd_return_net']:+.2%} | "
                      f"net@Rs{COST_POSITION_SENSITIVITY:,.0f} "
                      f"{v['mean_fwd_return_net_small']:+.2%}  "
                      f"({v['cohorts']} cohorts){tag}")
        print(f"  Ladder monotonic (5>=...>=0): {summary['ladder_monotonic']}")
        sp = summary["spread_high_minus_low"]
        if sp["point"] is not None:
            band = (f"  95% CI {sp['ci95'][0]:+.2%}..{sp['ci95'][1]:+.2%}"
                    if sp["ci95"] else "  (band forms at 2 cohorts)")
            print(f"  High-low spread mean(4,5)-mean(0,1): {sp['point']:+.2%}{band}")
        print("  Survivorship note: n_missing upward-biases these; see caveats.")

    print("\nBacktest complete.")


if __name__ == "__main__":
    main()
