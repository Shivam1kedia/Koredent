"""
macro_read.py — the READ side of the macro series. (Sprint 16)

macro_fetch.py has written macro_series.json every weekday since 2026-07-27 and
NOTHING has ever read it: ~264 Tavily credits a month plus a Gemini call per
field, producing 60KB that no consumer opened. This module is the consumer.

Split from macro_fetch on purpose. macro_fetch imports requests and google-genai
and calls sys.exit(1) at import when genai is missing, so any module that reads
the series through it inherits a fatal network dependency to look at a local
JSON file. This module imports nothing beyond the standard library and never
exits. macro_fetch imports operative_value FROM HERE, so the definition the
writer's falsification report uses and the definition production reads are the
same object, not two copies free to drift.

WHAT IS AND IS NOT WIRED

  WIRED: portfolio_tracker.get_india_rfr() — the risk-free rate in Sharpe,
  Sortino, Treynor, Jensen and CAPM. Unambiguously correct there: it is the
  actual contemporaneous risk-free rate those formulas ask for, it changes no
  stored score, and it is bounded and falls back.

  NOT WIRED, deliberately: deep_metrics.INDIA_10Y_BOND_RATE. That constant
  feeds graham_earnings_yield_spread, which feeds the Graham framework, which
  feeds `score`. Making it live would re-score ~4,500 stocks whenever the G-Sec
  moved 4bp, break archive comparability, and require a SCHEMA_VERSION bump for
  what is noise, not signal. It also sits in a Gordon-growth denominator
  (deep_metrics BM1) where r -> g makes intrinsic value explode. The scoring
  constant stays frozen and dated; see the block at deep_metrics.py:22. The
  live value is carried alongside for MONITORING only, via rate_monitor().

  EXPOSED, not yet consumed: the four tax fields (Sprints 18/20/21) and the RBI
  CPI projection (Sprint 18's real-return work). Exposing a reader is not a
  decision to use the number.

FRESHNESS. operative_value medians the last few readings, which kills a single
wild parse — ltcg_pct came back 10.0 on 2026-09-18 against 12.5 on the four
days before it, and the median absorbed it without a special case. What the
median cannot see is the whole series being old, so every accessor here also
checks the newest reading against TODAY and returns STALE rather than a
confidently wrong number from three weeks ago.

STATUS IS NEVER SILENT. Every accessor returns (value, status). A caller that
wants a plain number asks for a fallback explicitly. "The macro job has been
down for a week" and "the rate is 7.05%" must not look alike at the call site.
"""

import json
import os
from datetime import date, datetime
from statistics import median

# Resolved against this file, not the process cwd. portfolio_tracker runs from
# the repo root in Actions, Streamlit runs from the app directory, and a
# diagnostic script runs from wherever it was invoked.
SERIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "macro_series.json")
SCHEMA_VERSION = 1

# Default window for the rolling median. Median, not mean: the failure mode is
# one wild parse, not drift.
WINDOW = 5
MAX_SPAN_DAYS = 10

# The series is written on weekdays. A long weekend plus a market holiday is
# four calendar days, so seven means "the job has been failing for a week",
# not "it is Sunday".
MAX_AGE_DAYS = 7

# Plausibility bands, per field. These catch a SUSTAINED bad source, which the
# median cannot — if tradingeconomics starts serving the US 10Y under the India
# label, five consistent readings of 4.2 would pass the median and fail here.
# Wide on purpose: a band that fires on ordinary movement is a band that gets
# removed the first time it is inconvenient.
BANDS = {
    "india_10y_yield_pct": (4.0, 10.0),
    "rbi_cpi_projection_pct": (0.0, 12.0),
    "ltcg_pct": (0.0, 30.0),
    "stcg_pct": (0.0, 40.0),
    "ltcg_holding_months": (1.0, 60.0),
    "ltcg_exemption_inr": (0.0, 1_000_000.0),
}

# Fallbacks. Used ONLY when the series cannot answer, and every accessor says
# so in its status so a fallback is never mistaken for a reading.
FALLBACK_INDIA_RFR = 0.07        # India 10Y G-Sec, ~7.0% through 2026

_CACHE = {"readings": None, "mtime": None}


# ── Series I/O ────────────────────────────────────────────────────────────
def read_series(path=None):
    """Every reading in the series, or [] if it cannot be read.

    Lenient by design, unlike macro_fetch.load_series which exits on a schema
    mismatch. The writer must refuse to append to a file it does not
    understand; a reader must degrade to its fallback and let the caller keep
    working. Cached on mtime — 4,500 universe rows must not each open a 60KB
    JSON file.
    """
    p = path or SERIES_PATH
    try:
        mtime = os.path.getmtime(p)
    except OSError:
        return []
    if _CACHE["readings"] is not None and _CACHE["mtime"] == mtime:
        return _CACHE["readings"]
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    if data.get("schema") != SCHEMA_VERSION:
        return []
    rows = data.get("readings")
    rows = rows if isinstance(rows, list) else []
    _CACHE["readings"], _CACHE["mtime"] = rows, mtime
    return rows


def operative_value(readings, field_name, window=WINDOW,
                    max_span_days=MAX_SPAN_DAYS):
    """Rolling median of the last `window` OK readings, within max_span_days.

    THE definition of an operative value, imported by macro_fetch so its
    falsification report and every production consumer agree.

    Median, not mean: the failure mode is one wild parse, not drift. RBI's
    projection is a step function with ~6 steps a year, so a 5-reading median
    lags a genuine step by 2-3 days and rejects everything else.

    Re-runs on the same date collapse to the LAST reading for that date, so a
    manual re-trigger does not double-weight a day.
    """
    by_date = {}
    for r in readings:
        if r.get("field") == field_name and r.get("status") == "ok" \
                and r.get("value") is not None:
            by_date[r["date"]] = r["value"]

    if not by_date:
        return None, "INSUFFICIENT"

    dates = sorted(by_date)[-window:]
    newest = datetime.strptime(dates[-1], "%Y-%m-%d").date()
    oldest = datetime.strptime(dates[0], "%Y-%m-%d").date()
    if (newest - oldest).days > max_span_days:
        dates = [d for d in dates
                 if (newest - datetime.strptime(d, "%Y-%m-%d").date()).days
                 <= max_span_days]

    if len(dates) < 3:
        return None, "INSUFFICIENT"
    return median(by_date[d] for d in dates), "ok"


# ── The accessor every consumer uses ──────────────────────────────────────
def operative(field_name, as_of=None, max_age_days=MAX_AGE_DAYS, path=None):
    """(value, status) for one field, with freshness and band checks applied.

    status is one of:
      ok            a usable operative value
      UNAVAILABLE   no series file, unreadable, or a schema this reader
                    does not know
      INSUFFICIENT  fewer than three OK readings inside the median window
      STALE         newest OK reading is older than max_age_days
      OUT_OF_BAND   the median is outside the field's plausibility band —
                    a sustained bad source, which the median cannot catch

    A value is returned ONLY with status "ok". Every other status returns None,
    so a caller that ignores the status gets a None it must handle rather than
    a number it will trust.
    """
    rows = read_series(path)
    if not rows:
        return None, "UNAVAILABLE"

    val, status = operative_value(rows, field_name)
    if status != "ok":
        return None, status

    ok_dates = [r["date"] for r in rows
                if r.get("field") == field_name and r.get("status") == "ok"
                and r.get("value") is not None]
    try:
        newest = datetime.strptime(max(ok_dates), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None, "UNAVAILABLE"
    ref = as_of or date.today()
    if (ref - newest).days > max_age_days:
        return None, "STALE"

    lo, hi = BANDS.get(field_name, (float("-inf"), float("inf")))
    if not (lo <= val <= hi):
        return None, "OUT_OF_BAND"

    return val, "ok"


# ── Typed accessors ───────────────────────────────────────────────────────
def india_rfr(fallback=FALLBACK_INDIA_RFR, as_of=None):
    """(rate_as_decimal, status). 0.0705 == 7.05%.

    THE risk-free rate for portfolio metrics. Falls back to the documented
    constant on any non-ok status, and the status says which. Never raises:
    a tracker run must not die because a JSON file is missing.
    """
    pct, status = operative("india_10y_yield_pct", as_of=as_of)
    if status != "ok" or pct is None:
        return float(fallback), status
    return round(float(pct) / 100.0, 6), "ok"


def cpi_projection(as_of=None):
    """(rbi_cpi_projection_pct, status). Exposed for Sprint 18's real-return
    work; no consumer yet.

    Read the status before using this one. rbi.org.in is a hard source to
    extract a single forward projection from — the field's own readings have
    ranged 2.1 to 5.4 inside a week, which is extraction noise, not RBI
    revising its projection three times. The median damps it; it does not make
    it a good series. Falsify before wiring.
    """
    return operative("rbi_cpi_projection_pct", as_of=as_of)


def tax_params(as_of=None):
    """{field: (value, status)} for the four capital-gains parameters.

    Exposed for Sprints 18, 20 and 21. Fetched daily since 2026-07-27 and
    unused since. The three stable ones (stcg 20%, holding 12 months,
    exemption Rs 1.25 lakh) have been unanimous; ltcg_pct has one bad parse in
    the record, which is exactly what the median is for.
    """
    return {f: operative(f, as_of=as_of) for f in
            ("ltcg_pct", "stcg_pct", "ltcg_holding_months", "ltcg_exemption_inr")}


def rate_monitor(frozen_pct, as_of=None):
    """Compare the live 10Y against a frozen SCORING constant. Monitoring only.

    Returns {frozen, live, status, drift_bp, exceeds}. `exceeds` is True when
    the live rate has moved more than REEXAMINE_BP from the frozen constant,
    which is not a failure and must not auto-update anything — it is the
    trigger to re-examine scoring sensitivity as its own decision, with a
    SCHEMA_VERSION bump and a reconcile run, which is a sprint and not a
    config change.
    """
    live, status = operative("india_10y_yield_pct", as_of=as_of)
    out = {"frozen_pct": frozen_pct, "live_pct": live, "status": status,
           "drift_bp": None, "exceeds": False}
    if status == "ok" and live is not None:
        out["drift_bp"] = round((live - frozen_pct) * 100, 1)
        out["exceeds"] = abs(out["drift_bp"]) > REEXAMINE_BP
    return out


# Pre-registered 2026-09-18, with the current drift already on screen and
# stated rather than hidden: live operative 7.055 against a frozen 7.0 is
# 5.5bp. 100bp is the point at which the frozen scoring constant stops being
# a fair description of the rate environment the Graham spread assumes.
REEXAMINE_BP = 100.0


def status_report(as_of=None):
    """Every exposed field with its operative value and status. For preflight,
    for a diagnostic run, and for any UI that wants to show what the system is
    actually using."""
    fields = ["india_10y_yield_pct", "rbi_cpi_projection_pct", "ltcg_pct",
              "stcg_pct", "ltcg_holding_months", "ltcg_exemption_inr"]
    return {f: operative(f, as_of=as_of) for f in fields}


if __name__ == "__main__":
    rows = read_series()
    print(f"{SERIES_PATH}: {len(rows)} readings\n")
    for f, (v, s) in status_report().items():
        print(f"  {f:<28} {str(v):>12}   [{s}]")
    r, s = india_rfr()
    print(f"\n  india_rfr() -> {r} ({r*100:.3f}%)   [{s}]")
    print(f"  rate_monitor(7.0) -> {rate_monitor(7.0)}")
