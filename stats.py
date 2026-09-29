"""
stats.py — descriptive statistics about the SCORER, not its performance.

Half A of the Sprint 12 statistics layer. Everything here is computable today,
with zero forward returns, because it describes the shape of the universe as the
scorer sees it right now:

  - flag base rates          how often each of the 5 frameworks passes
  - phi orthogonality        whether the flags measure different things
  - score distribution       how the universe spreads across 0..5
  - trajectory cliff         the pile of stocks at exactly 7 (gate is >= 8)

HONESTY BOUNDARY: none of this says a flag PREDICTS anything. A base rate is not
a win rate. Orthogonality means "independent", not "useful". Those claims need
forward returns, which do not exist yet. Everything here is labelled as
descriptive so the "Does It Work?" page cannot be misread as a performance claim.

Single source of truth for the flag set: FLAGS. If a sixth framework is added,
add it here and every stat updates.
"""

import math
import pandas as pd

FLAGS = ["graham_pass", "greenblatt_pass", "dorsey_pass",
         "trajectory_pass", "lynch_pass"]

# Display names, kept out of the app so the page and the LLM agree on labels.
FLAG_LABELS = {
    "graham_pass": "Graham",
    "greenblatt_pass": "Greenblatt",
    "dorsey_pass": "Dorsey+Buffett",
    "trajectory_pass": "Trajectory",
    "lynch_pass": "Lynch",
}

TRAJECTORY_GATE = 8   # deep_metrics: trajectory_pass = trajectory_score >= 8


def _as_bool(series):
    """CSV round-trips booleans as strings ('True'/'False') or 0/1. Coerce."""
    if series.dtype == bool:
        return series.fillna(False)
    return (series.astype(str).str.strip().str.lower()
            .isin(["true", "1", "1.0", "yes"]))


def _phi(a, b):
    """Phi coefficient (Pearson correlation between two booleans).

    phi = (n11*n00 - n10*n01) / sqrt(row/col marginals). Range -1..1.
    Returns None if either variable is constant (marginal is zero -> undefined).
    """
    n11 = int((a & b).sum())
    n10 = int((a & ~b).sum())
    n01 = int((~a & b).sum())
    n00 = int((~a & ~b).sum())
    num = n11 * n00 - n10 * n01
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    if den == 0:
        return None
    return num / den


def compute_universe_stats(universe_df):
    """Descriptive stats over the full scored universe.

    Parameters
    ----------
    universe_df : DataFrame with the 5 *_pass flags, `score`, `quality_pass`,
                  `trajectory_score`. Missing columns degrade gracefully — the
                  stat that needs them is omitted, never faked.

    Returns a dict ready for the "Does It Work?" page and LLM context.
    """
    df = universe_df
    n = len(df)
    out = {"n_universe": n, "descriptive_only": True}
    if n == 0:
        return out

    present = [f for f in FLAGS if f in df.columns]
    bools = {f: _as_bool(df[f]) for f in present}

    # ── 1. Base rates ────────────────────────────────────────────────────
    base = {}
    for f in present:
        base[f] = {
            "label": FLAG_LABELS.get(f, f),
            "pass_rate": round(float(bools[f].mean()), 4),
            "pass_count": int(bools[f].sum()),
        }
    out["base_rates"] = dict(sorted(base.items(),
                                    key=lambda kv: kv[1]["pass_rate"]))
    if base:
        rates = [v["pass_rate"] for v in base.values() if v["pass_rate"] > 0]
        if rates:
            out["base_rate_spread"] = {
                "min": round(min(rates), 4),
                "max": round(max(rates), 4),
                "ratio": round(max(rates) / min(rates), 1),
                "rarest": min(base, key=lambda k: base[k]["pass_rate"]),
                "commonest": max(base, key=lambda k: base[k]["pass_rate"]),
            }

    # ── 2. Phi orthogonality ─────────────────────────────────────────────
    # rare AND independent is the signature of a signal the others miss.
    phi = {}
    max_abs = {}
    for i, fi in enumerate(present):
        for fj in present[i + 1:]:
            p = _phi(bools[fi], bools[fj])
            phi[f"{fi}|{fj}"] = None if p is None else round(p, 3)
    for f in present:
        others = []
        for f2 in present:
            if f2 == f:
                continue
            key = f"{f}|{f2}" if f"{f}|{f2}" in phi else f"{f2}|{f}"
            v = phi.get(key)
            if v is not None:
                others.append(abs(v))
        max_abs[f] = round(max(others), 3) if others else None
    out["phi_matrix"] = phi
    out["phi_max_abs"] = dict(sorted(
        max_abs.items(),
        key=lambda kv: (kv[1] is None, kv[1])))
    # the most orthogonal flag: lowest max|phi| among those defined
    _defined = {k: v for k, v in max_abs.items() if v is not None}
    if _defined:
        out["most_orthogonal"] = min(_defined, key=_defined.get)

    # The real story lives in the PAIRS, not per-flag maxima. If three flags
    # travel together (high |phi|), a 3/5 that passes all three is a weaker
    # signal than a 3/5 spread across independent flags — they are ~1.5 votes,
    # not 3. Surface the most- and least-entangled pairs by name so the page
    # leads with the finding instead of burying it in the matrix.
    _pairs = [(k, v) for k, v in phi.items() if v is not None]
    if _pairs:
        _by_abs = sorted(_pairs, key=lambda kv: abs(kv[1]))
        def _pair_dict(key, val):
            fi, fj = key.split("|")
            return {
                "pair": key,
                "labels": [FLAG_LABELS.get(fi, fi), FLAG_LABELS.get(fj, fj)],
                "phi": val,
            }
        out["least_orthogonal_pair"] = _pair_dict(*_by_abs[-1])   # most entangled
        out["most_orthogonal_pair"] = _pair_dict(*_by_abs[0])     # most independent
        # the entangled cluster: every pair above a redundancy-worth-noting bar
        CLUSTER_PHI = 0.30
        out["correlated_cluster"] = [
            _pair_dict(k, v) for k, v in
            sorted(_pairs, key=lambda kv: abs(kv[1]), reverse=True)
            if abs(v) >= CLUSTER_PHI
        ]

    # ── 3. Score distribution ────────────────────────────────────────────
    if "score" in df.columns:
        sc = pd.to_numeric(df["score"], errors="coerce").dropna().astype(int)
        dist = {int(k): int(v) for k, v in sc.value_counts().sort_index().items()}
        # ensure 0..5 all present
        out["score_distribution"] = {s: dist.get(s, 0) for s in range(6)}
        out["score_pct"] = {s: round(dist.get(s, 0) / n, 4) for s in range(6)}

    if "quality_pass" in df.columns:
        qp = _as_bool(df["quality_pass"])
        out["quality_gate"] = {
            "pass": int(qp.sum()),
            "fail": int((~qp).sum()),
            "pass_rate": round(float(qp.mean()), 4),
        }

    # ── 4. Trajectory cliff ──────────────────────────────────────────────
    # The gate is a hard >= 8. If a large mass sits at exactly 7, the boundary
    # is a cliff, not a slope — a design fact worth showing, not hiding.
    if "trajectory_score" in df.columns:
        ts = pd.to_numeric(df["trajectory_score"], errors="coerce").dropna().astype(int)
        hist = {int(k): int(v) for k, v in ts.value_counts().sort_index().items()}
        out["trajectory_hist"] = {s: hist.get(s, 0) for s in range(11)}
        _at_boundary = hist.get(TRAJECTORY_GATE - 1, 0)
        _passing = int((ts >= TRAJECTORY_GATE).sum())
        out["trajectory_cliff"] = {
            "gate": TRAJECTORY_GATE,
            "at_boundary": _at_boundary,               # stuck at gate-1
            "boundary_score": TRAJECTORY_GATE - 1,
            "passing": _passing,
            # how big the just-missed pile is relative to those who passed
            "boundary_vs_passing_ratio": (round(_at_boundary / _passing, 2)
                                          if _passing else None),
        }

    return out


# ══════════════════════════════════════════════════════════════════════════
# HALF B — GOAL PROJECTION AS A DISTRIBUTION, NOT A LINE
# ══════════════════════════════════════════════════════════════════════════
# The old goal projection drew ONE curve at a fixed 12% CAGR and reported one
# number: "you'll have Rs X." That is false precision — the same sin the risk
# ranges fixed. Half B replaces the certainty (not the function): keep the
# deterministic median line, add a probability FAN around it and P(hit target).
#
# HONESTY, built structurally:
#   - Block bootstrap (12-month blocks) preserves crash CLUSTERING and sequence
#     risk. Shuffling single months would manufacture a tidy near-normal outcome
#     that hides the chance of a lost decade. Bad months come in runs; we keep
#     the runs.
#   - Max-available history (Sensex reaches ~1997, incl. 2000/2008/2020) because
#     the typical user projects 25 years. A 25-year path sampled from 15 BULL
#     years is a brochure. The sample must contain the crises a 25-year holder
#     could actually live through.
#   - The distribution is the INDEX's, not Kordent's. Every output is labelled
#     `benchmark: <name>` and `is_proxy: True`. "82% chance" always means "IF
#     your portfolio behaves like the index". Swappable for Kordent's own
#     realised returns once they exist.
#   - Past returns as future distribution is an OPTIMISTIC assumption (India
#     equity history is bull-heavy). Output carries `bias: "optimistic_ceiling"`.

import datetime as _dt

_NIFTY_CACHE = {"returns": None, "meta": None}


def fetch_index_monthly_returns(force=False):
    """~Max-available monthly returns for an Indian large-cap index, as a list
    of floats (0.02 == +2%). Tries Nifty 50, falls back to Sensex for the longer
    tail. Cached per process. Returns (returns_list, meta_dict).

    meta: {benchmark, start, end, n_months}. On total failure returns (None, err).
    Never raises.
    """
    if _NIFTY_CACHE["returns"] is not None and not force:
        return _NIFTY_CACHE["returns"], _NIFTY_CACHE["meta"]

    import yfinance as yf
    # (ticker, human name). Nifty first (what users know); Sensex reaches further
    # back on Yahoo and is a near-perfect large-cap proxy for the pre-2007 tail.
    for ticker, name in (("^NSEI", "Nifty 50"), ("^BSESN", "BSE Sensex")):
        try:
            hist = yf.download(ticker, start="1990-01-01", interval="1mo",
                               progress=False, auto_adjust=True)
            if hist is None or len(hist) < 60:
                continue
            close = hist["Close"]
            # yfinance 1.x hands back a DataFrame even for one ticker
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            rets = close.pct_change(fill_method=None).dropna()
            rets = rets[(rets > -0.6) & (rets < 0.6)]  # drop data-glitch spikes
            if len(rets) < 60:
                continue
            meta = {
                "benchmark": name,
                "start": str(rets.index[0].date()),
                "end": str(rets.index[-1].date()),
                "n_months": int(len(rets)),
                "is_proxy": True,
                "bias": "optimistic_ceiling",
            }
            out = [float(x) for x in rets.tolist()]
            _NIFTY_CACHE["returns"], _NIFTY_CACHE["meta"] = out, meta
            return out, meta
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    return None, {"error": "no index history available"}


def project_goal_distribution(current_value, sip_monthly, target_amount,
                              months_remaining, monthly_returns=None,
                              n_paths=2000, block=12, seed=20260712):
    """Monte Carlo goal projection using a block bootstrap of real index months.

    Parameters
    ----------
    current_value    : starting corpus (INR)
    sip_monthly      : monthly SIP contribution (INR)
    target_amount    : goal (INR); may be None/0 -> P(hit) omitted
    months_remaining : horizon in months
    monthly_returns  : list of historical monthly returns. If None, fetched.
    n_paths          : number of simulated futures
    block            : bootstrap block length in months (12 keeps crash runs)

    Returns a dict with percentile terminal values, P(hit target), a percentile
    fan for charting, and the benchmark label/caveat. None on unusable inputs.
    """
    import numpy as np

    if months_remaining is None or months_remaining <= 0:
        return None

    meta = None
    if monthly_returns is None:
        monthly_returns, meta = fetch_index_monthly_returns()
    if not monthly_returns or len(monthly_returns) < block * 2:
        return None
    if meta is None:
        meta = {"benchmark": "index", "is_proxy": True,
                "bias": "optimistic_ceiling"}

    r = np.asarray(monthly_returns, dtype=float)
    H = int(months_remaining)
    rng = np.random.default_rng(seed)

    # Build each path from contiguous blocks so autocorrelation/crash-clustering
    # survives. n_blocks blocks of `block` months, trimmed to H.
    n_blocks = int(np.ceil(H / block))
    n_start = len(r) - block  # valid block start indices: 0..n_start

    # sample matrix of block start indices: (n_paths, n_blocks)
    starts = rng.integers(0, n_start + 1, size=(n_paths, n_blocks))
    # expand to month-level return matrix (n_paths, n_blocks*block) then trim
    offsets = np.arange(block)
    idx = starts[:, :, None] + offsets[None, None, :]      # (paths, blocks, block)
    idx = idx.reshape(n_paths, n_blocks * block)[:, :H]     # (paths, H)
    path_returns = r[idx]                                   # (paths, H)

    # Compound: each month multiply corpus by (1+r), then add SIP at month end.
    corpus = np.full(n_paths, float(current_value))
    for m in range(H):
        corpus = corpus * (1.0 + path_returns[:, m]) + sip_monthly
    terminal = corpus

    pct = {p: float(np.percentile(terminal, p)) for p in (10, 25, 50, 75, 90)}

    result = {
        "benchmark": meta.get("benchmark", "index"),
        "is_proxy": True,
        "bias": "optimistic_ceiling",
        "history_start": meta.get("start"),
        "history_end": meta.get("end"),
        "history_months": meta.get("n_months"),
        "n_paths": n_paths,
        "horizon_months": H,
        "block_months": block,
        "terminal_p10": round(pct[10]),
        "terminal_p25": round(pct[25]),
        "terminal_p50": round(pct[50]),
        "terminal_p75": round(pct[75]),
        "terminal_p90": round(pct[90]),
    }

    if target_amount and target_amount > 0:
        result["target_amount"] = float(target_amount)
        result["prob_hit_target"] = round(float((terminal >= target_amount).mean()), 4)

    # Independent-window honesty: how many non-overlapping H-month windows the
    # history actually contains. Few windows => the fan is stitched, not observed.
    result["independent_windows"] = max(1, len(r) // H)

    # Force the caveat to ride WITH the probability. A 24% that came from zero
    # complete historical windows must never be read as a forecast. This tag is
    # inseparable from prob_hit_target wherever it is shown or handed to the LLM.
    _w = result["independent_windows"]
    if _w >= 10:
        result["confidence"] = "well-sampled"
        result["confidence_note"] = (
            f"Based on {_w} independent {H // 12}-year periods in the history.")
    elif _w >= 3:
        result["confidence"] = "thinly-sampled"
        result["confidence_note"] = (
            f"Based on only {_w} independent {H // 12}-year periods — a wide "
            f"band is honest, not pessimistic.")
    else:
        result["confidence"] = "stitched"
        result["confidence_note"] = (
            f"No complete {H // 12}-year period exists in the available history "
            f"({len(r)} months). This range is stitched from shorter blocks — "
            f"illustrative, not a forecast.")

    # Percentile fan over time for charting: p10/p50/p90 corpus at each month.
    # Recompute cumulative paths cheaply for the fan (reuse path_returns).
    fan_corpus = np.full(n_paths, float(current_value))
    fan = {"p10": [], "p50": [], "p90": []}
    step = max(1, H // 60)  # cap at ~60 points for a clean chart
    for m in range(H):
        fan_corpus = fan_corpus * (1.0 + path_returns[:, m]) + sip_monthly
        if (m + 1) % step == 0 or m == H - 1:
            fan["p10"].append(round(float(np.percentile(fan_corpus, 10))))
            fan["p50"].append(round(float(np.percentile(fan_corpus, 50))))
            fan["p90"].append(round(float(np.percentile(fan_corpus, 90))))
    result["fan"] = fan
    result["fan_step_months"] = step

    return result
