"""
axis_scores.py — W0.3 relative axis scoring (Quality / Growth / Price / Safety).

TWO layers, by design:
  - ABSOLUTE floor (elsewhere): the quality gate + the book-ramp 5-vector answer
    "is this honestly good enough, and has its thesis drifted." India-calibrated.
  - RELATIVE profile (here): "among what's available, how does this stock rank on each
    axis for a user's demands." Selection is comparison, so this layer is relative —
    but graded against a FROZEN Indian reference so a stock's score moves only when its
    OWN metric moves, never because the universe moved.

Method:
  build : from a CLEAN (schema_version>=2) snapshot, freeze per-metric CDF knots +
          within-axis INDEPENDENCE weights (1 / sum|corr| over same-axis metrics, on the
          oriented rank basis the axis audit validated). Write axis_reference.json, commit it.
  grade : for each stock, empirical-CDF each metric against the frozen knots (oriented so
          higher=better), then per-axis weighted mean. Missing metrics are DROPPED and the
          axis's weights renormalized over what's present (missing != zero). All-missing
          axis -> None.

Only CONTINUOUS metrics are scored here; boolean axis checks live in the absolute 5-vector.

Usage:
  py axis_scores.py build [clean_universe.csv]     # -> axis_reference.json  (run on v>=2)
  py axis_scores.py grade [universe.csv]           # -> prints axis-score summary
"""
import sys, json
import numpy as np
import pandas as pd

REF_PATH = "axis_reference.json"
KNOTS = 200                      # CDF resolution: 201 quantile knots per metric

# metric -> (axis, direction). direction: up | down | inv (down/inv flip to "higher=better").
# Kept in sync with the PCA-validated axis table (post v1-audit moves).
AXIS_MAP = {
    # QUALITY
    "graham_eps_cv": ("Quality", "down"), "greenblatt_roic": ("Quality", "up"),
    "dorsey_roic": ("Quality", "up"), "dorsey_roa": ("Quality", "up"),
    "buffett_one_dollar_test": ("Quality", "up"), "dividend_consecutive_years": ("Quality", "up"),
    "roe": ("Quality", "up"), "profit_margin": ("Quality", "up"),
    # (dorsey_share_dilution_pct rejoins Quality once its all-NaN population bug is fixed)
    # GROWTH
    "graham_eps_growth_pct_4y": ("Growth", "up"), "lynch_growth_acceleration": ("Growth", "up"),
    "greenblatt_roic_trend": ("Growth", "up"), "revenue_cagr_3y": ("Growth", "up"),
    "ni_cagr_3y": ("Growth", "up"),
    # PRICE
    "graham_ncav_ratio": ("Price", "down"), "graham_price_to_ntav": ("Price", "down"),
    "graham_pe_3y_avg": ("Price", "down"), "graham_pe_pb_composite": ("Price", "down"),
    "graham_earnings_yield_spread": ("Price", "up"), "graham_margin_of_safety_pct": ("Price", "up"),
    "greenblatt_earnings_yield": ("Price", "up"), "lynch_peg": ("Price", "down"),
    "lynch_peg_adjusted": ("Price", "up"), "lynch_cash_adjusted_pe": ("Price", "down"),
    "dorsey_cash_return": ("Price", "up"), "buffett_margin_of_safety_pct": ("Price", "up"),
    "pe": ("Price", "down"), "pb": ("Price", "down"),
    # SAFETY  (many inv: raw metric measures risk)
    "current_ratio": ("Safety", "up"), "dorsey_fcf_margin": ("Safety", "up"),
    "lynch_net_cash_per_share": ("Safety", "up"), "dorsey_financial_leverage": ("Safety", "down"),
    "dorsey_interest_coverage": ("Safety", "up"), "dorsey_quick_ratio": ("Safety", "up"),
    "schilit_accruals_ratio": ("Safety", "inv"), "schilit_cfo_ni_ratio": ("Safety", "up"),
    "schilit_fcf_ni_ratio": ("Safety", "up"), "mulford_ecm": ("Safety", "up"),
    "mulford_ecm_trend": ("Safety", "up"), "mulford_cash_margin": ("Safety", "up"),
    "mulford_ocf_oi_ratio": ("Safety", "up"), "schilit_dso": ("Safety", "inv"),
    "schilit_ar_revenue_divergence": ("Safety", "inv"), "schilit_dsi": ("Safety", "inv"),
    "schilit_inventory_revenue_div": ("Safety", "inv"), "schilit_wc_cffo_pct": ("Safety", "inv"),
    "schilit_leverage_trend": ("Safety", "inv"), "goodwill_pct": ("Safety", "inv"),
}
AXES = ["Quality", "Growth", "Price", "Safety"]
MISSING_DROP = 0.50


def _oriented_rank(s, direction):
    """pct rank oriented so higher=better; missing -> neutral 0.5 (for corr only)."""
    r = pd.to_numeric(s, errors="coerce").rank(pct=True)
    if direction in ("down", "inv"):
        r = 1 - r
    return r.fillna(0.5)


def build(csv):
    df = pd.read_csv(csv)
    n = len(df)
    print(f"Building reference from {n} rows ({csv})\n")

    present = {}
    for name, (axis, direction) in AXIS_MAP.items():
        if name not in df.columns:
            print(f"  [skip] {name}: absent"); continue
        s = pd.to_numeric(df[name], errors="coerce")
        m = s.isna().mean()
        if m > MISSING_DROP:
            print(f"  [drop] {name}: {m:.0%} missing"); continue
        if s.dropna().nunique() < 5:
            print(f"  [drop] {name}: too few distinct values"); continue
        present[name] = (axis, direction, s)

    # ---- CDF knots per metric (quantiles of the RAW oriented-later distribution) ----
    levels = np.linspace(0.0, 1.0, KNOTS + 1)
    ref = {"knots_levels": levels.tolist(), "metrics": {}, "axis_weights": {}}
    for name, (axis, direction, s) in present.items():
        knots = np.nanquantile(s.dropna().values.astype(float), levels)
        # de-duplicate flat regions so np.interp stays monotone
        knots = np.maximum.accumulate(knots)
        ref["metrics"][name] = {"axis": axis, "direction": direction,
                                "knots": [float(x) for x in knots]}

    # ---- within-axis INDEPENDENCE weights from oriented-rank correlation ----
    for axis in AXES:
        names = [n for n in present if present[n][0] == axis]
        if not names:
            continue
        R = np.column_stack([_oriented_rank(present[n][2], present[n][1]).values for n in names])
        C = np.corrcoef(R, rowvar=False)
        if C.ndim == 0:                       # single metric
            C = np.array([[1.0]])
        absrowsum = np.abs(C).sum(axis=1)     # includes self=1; high => redundant
        w = 1.0 / np.where(absrowsum > 0, absrowsum, 1.0)
        w = w / w.sum()                       # normalize within axis
        ref["axis_weights"][axis] = {n: float(wi) for n, wi in zip(names, w)}
        # effective number of independent metrics = 1 / sum(w^2)  (report only)
        eff = 1.0 / np.sum((w) ** 2)
        print(f"  {axis:<8} {len(names)} metrics -> eff. independent ≈ {eff:.1f}")
        for n, wi in sorted(zip(names, w), key=lambda x: -x[1]):
            print(f"       w={wi:.3f}  {n}")

    with open(REF_PATH, "w") as f:
        json.dump(ref, f, indent=1)
    print(f"\nWrote {REF_PATH}  ({len(ref['metrics'])} metrics)."
          f"\nNOTE: freeze this on a schema_version>=2 file, then COMMIT it — it is the"
          f"\nfixed standard every future score is graded against.")


def _cdf(value, knots, levels, direction):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    p = float(np.interp(float(value), knots, levels))  # clipped to [0,1] at the ends
    if direction in ("down", "inv"):
        p = 1.0 - p
    return min(1.0, max(0.0, p))


def grade_row(row, ref):
    """row: dict-like. -> {quality_axis, growth_axis, price_axis, safety_axis} in [0,1] or None."""
    levels = np.array(ref["knots_levels"])
    out = {}
    for axis in AXES:
        weights = ref["axis_weights"].get(axis, {})
        num, wsum = 0.0, 0.0
        for name, w in weights.items():
            meta = ref["metrics"].get(name)
            if not meta:
                continue
            p = _cdf(row.get(name), np.array(meta["knots"]), levels, meta["direction"])
            if p is None:
                continue                      # missing -> drop, renormalize over present
            num += w * p
            wsum += w
        out[f"{axis.lower()}_axis"] = round(num / wsum, 4) if wsum > 0 else None
    return out


def grade(csv):
    with open(REF_PATH) as f:
        ref = json.load(f)
    df = pd.read_csv(csv)
    scores = [grade_row({k: (None if (isinstance(v, float) and np.isnan(v)) else v)
                         for k, v in r.items()}, ref) for _, r in df.iterrows()]
    sc = pd.DataFrame(scores)
    print(f"Graded {len(df)} rows against {REF_PATH}\n")
    print(sc.describe().round(3).T[["count", "mean", "std", "min", "50%", "max"]])
    print("\n(Each axis score is a [0,1] position vs the frozen Indian reference, "
          "independence-weighted. Show alongside the absolute gate + 5-vector, never alone.)")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "build"
    path = sys.argv[2] if len(sys.argv) > 2 else "universe_scored.csv"
    (build if mode == "build" else grade)(path)
