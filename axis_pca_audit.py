"""
axis_pca_audit.py — does the Quality/Growth/Price/Safety carving match the DATA's
own structure? Standalone diagnostic. Reads universe_scored.csv, does NOT touch the
pipeline. Cross-sectional only (one snapshot) — no forward returns anywhere.

Method (revised after the v1 run — financial cross-sections are wildly skewed):
  - only CONTINUOUS, monotonic metrics enter PCA (bool/cat/band excluded — listed below)
  - RANK-transform each metric (pct rank) -> Spearman correlation. Bounded, so a few
    PE=5000 outliers can't make their own component; monotonic-invariant, so a %-vs-fraction
    unit mismatch (e.g. profit_margin) can't flip a metric mid-column. No winsorizing needed.
  - drop any metric >50% missing (imputing that is fiction)
  - missing -> neutral median rank (0.5): pulls toward average, never invents signal
  - ORIENT every metric so higher = better on its axis (flip 'down'/'inv')
  - PCA via SVD on the rank matrix, varimax-rotate the retained factors

  v1-run findings already folded into the axis map below:
    dorsey_fcf_margin  Quality->Safety (loaded on cash-conversion factor)
    lynch_net_cash_per_share  Price->Safety (loaded on liquidity factor)
    graham_net_cash / dorsey_share_dilution_pct  -> dropped (absolute-size / all-NaN)

Read the output top-down:
  1. WITHIN vs ACROSS axis correlation — the headline. within >> across = real axes.
  2. Does GROWTH separate from QUALITY? (the 4th-axis question)
  3. Scree — how many factors the data actually has (~4-8 expected).
  4. Rotated factors — which axis each factor is made of; cross-loaders flagged.

Usage:  py axis_pca_audit.py [universe_scored.csv]
"""
import sys
import numpy as np
import pandas as pd

CSV = sys.argv[1] if len(sys.argv) > 1 else "universe_scored.csv"

# ── Axis map: metric -> (axis, direction). Only CONTINUOUS metrics here. ──
# direction: 'up' higher-better | 'down' lower-better | 'inv' raw=risk (flip like down)
AXIS_MAP = {
    # QUALITY
    "graham_eps_cv":               ("Quality", "down"),
    "greenblatt_roic":             ("Quality", "up"),
    "dorsey_roic":                 ("Quality", "up"),
    "dorsey_roa":                  ("Quality", "up"),
    "buffett_one_dollar_test":     ("Quality", "up"),
    "dividend_consecutive_years":  ("Quality", "up"),
    "roe":                         ("Quality", "up"),
    "profit_margin":               ("Quality", "up"),
    # GROWTH
    "graham_eps_growth_pct_4y":    ("Growth", "up"),
    "lynch_growth_acceleration":   ("Growth", "up"),
    "greenblatt_roic_trend":       ("Growth", "up"),   # borderline Quality
    "revenue_cagr_3y":             ("Growth", "up"),
    "ni_cagr_3y":                  ("Growth", "up"),
    # PRICE  (higher = cheaper / more margin)
    "graham_ncav_ratio":           ("Price", "down"),
    "graham_price_to_ntav":        ("Price", "down"),
    "graham_pe_3y_avg":            ("Price", "down"),
    "graham_pe_pb_composite":      ("Price", "down"),
    "graham_earnings_yield_spread":("Price", "up"),
    "graham_margin_of_safety_pct": ("Price", "up"),
    "greenblatt_earnings_yield":   ("Price", "up"),
    "lynch_peg":                   ("Price", "down"),
    "lynch_peg_adjusted":          ("Price", "up"),
    "lynch_cash_adjusted_pe":      ("Price", "down"),
    "dorsey_cash_return":          ("Price", "up"),
    "buffett_margin_of_safety_pct":("Price", "up"),
    "pe":                          ("Price", "down"),
    "pb":                          ("Price", "down"),
    # SAFETY  (higher = safer; 'inv' raw measures risk)
    "current_ratio":               ("Safety", "up"),
    "dorsey_fcf_margin":           ("Safety", "up"),   # moved from Quality (cash-conversion)
    "lynch_net_cash_per_share":    ("Safety", "up"),   # moved from Price (liquidity)
    "dorsey_financial_leverage":   ("Safety", "down"),
    "dorsey_interest_coverage":    ("Safety", "up"),
    "dorsey_quick_ratio":          ("Safety", "up"),
    "schilit_accruals_ratio":      ("Safety", "inv"),
    "schilit_cfo_ni_ratio":        ("Safety", "up"),
    "schilit_fcf_ni_ratio":        ("Safety", "up"),
    "mulford_ecm":                 ("Safety", "up"),
    "mulford_ecm_trend":           ("Safety", "up"),
    "mulford_cash_margin":         ("Safety", "up"),
    "mulford_ocf_oi_ratio":        ("Safety", "up"),
    "schilit_dso":                 ("Safety", "inv"),
    "schilit_ar_revenue_divergence":("Safety", "inv"),
    "schilit_dsi":                 ("Safety", "inv"),
    "schilit_inventory_revenue_div":("Safety", "inv"),
    "schilit_wc_cffo_pct":         ("Safety", "inv"),
    "schilit_leverage_trend":      ("Safety", "inv"),
    "goodwill_pct":                ("Safety", "inv"),
}
# Excluded from PCA by nature (documented, not scored here):
#   bool: dorsey_roe_consistent, dorsey_pb_roe_signal, buffett_roe_unleveraged,
#         buffett_rational_allocation, dorsey_has_operating_profit, graham_deep_value_flag,
#         graham_ent_earnings_growing, buffett_value_creating_growth, graham_adequate_size,
#         graham_ltd_vs_nca, graham_earnings_stable_4y, dorsey_clean_balance_sheet,
#         dorsey_consistent_cfo, mulford_fcf_consistent, graham_ent_financial_pass, + inv bools
#   cat:  lynch_growth_flag, lynch_category, lynch_debt_healthy, mulford_lifecycle_stage
#   band: graham_payout_ratio, schilit_capex_depr_ratio

MISSING_DROP = 0.50      # drop metric if >50% NaN


def varimax(Phi, q=100, tol=1e-6):
    """Standard varimax rotation of a p x k loading matrix."""
    p, k = Phi.shape
    if k < 2:
        return Phi
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        diagsum = np.diag(np.sum(L ** 2, axis=0))
        u, s, vh = np.linalg.svd(Phi.T @ (L ** 3 - L @ diagsum / p))
        R = u @ vh
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R


def main():
    df = pd.read_csv(CSV)
    n_all = len(df)
    print(f"Loaded {n_all} rows x {df.shape[1]} cols from {CSV}\n")

    # ---- assemble, coerce, drop missing, RANK-transform, orient, neutral-fill ----
    cols, axes, miss = [], [], {}
    mat = []
    for name, (axis, direction) in AXIS_MAP.items():
        if name not in df.columns:
            print(f"  [skip] {name}: not in CSV")
            continue
        s = pd.to_numeric(df[name], errors="coerce")
        m = s.isna().mean()
        if m > MISSING_DROP:
            print(f"  [drop] {name}: {m:.0%} missing (>50%)")
            continue
        r = s.rank(pct=True)                       # Spearman: bounded, unit/outlier-proof
        if direction in ("down", "inv"):
            r = 1 - r
        r = r.fillna(0.5)                          # missing -> neutral median rank
        if r.std(ddof=0) == 0 or not np.isfinite(r.std(ddof=0)):
            print(f"  [drop] {name}: zero/nonfinite variance")
            continue
        z = (r - r.mean()) / r.std(ddof=0)
        cols.append(name); axes.append(axis); miss[name] = m
        mat.append(z.values)

    X = np.array(mat).T                     # n x p, oriented rank scores (Spearman basis)
    p = X.shape[1]
    axes = np.array(axes)
    print(f"\n{p} continuous metrics enter PCA across {len(set(axes))} axes "
          f"(Quality={np.sum(axes=='Quality')}, Growth={np.sum(axes=='Growth')}, "
          f"Price={np.sum(axes=='Price')}, Safety={np.sum(axes=='Safety')}).\n")

    C = np.corrcoef(X, rowvar=False)        # p x p correlation (metrics oriented same way)

    # ---- 1. WITHIN vs ACROSS axis correlation (the headline) ----
    def mean_abs(mask_pairs):
        vals = [abs(C[i, j]) for i, j in mask_pairs]
        return float(np.mean(vals)) if vals else float("nan")

    print("=" * 60)
    print("1. WITHIN vs ACROSS axis correlation  (within >> across = real axis)")
    print("=" * 60)
    for ax in ["Quality", "Growth", "Price", "Safety"]:
        idx = [i for i in range(p) if axes[i] == ax]
        within = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
        across = [(i, j) for i in idx for j in range(p) if axes[j] != ax]
        w, a = mean_abs(within), mean_abs(across)
        verdict = "coherent" if (w == w and a == a and w > a * 1.3) else "WEAK — inspect"
        print(f"  {ax:<8} within |r|={w:.3f}   across |r|={a:.3f}   -> {verdict}")

    # ---- 2. Growth vs Quality separation (the 4th-axis question) ----
    qi = [i for i in range(p) if axes[i] == "Quality"]
    gi = [i for i in range(p) if axes[i] == "Growth"]
    if qi and gi:
        qg = np.mean([abs(C[i, j]) for i in qi for j in gi])
        qq = mean_abs([(qi[a], qi[b]) for a in range(len(qi)) for b in range(a + 1, len(qi))])
        gg = mean_abs([(gi[a], gi[b]) for a in range(len(gi)) for b in range(a + 1, len(gi))])
        print("\n2. GROWTH vs QUALITY separation")
        print(f"   Quality-internal |r|={qq:.3f}  Growth-internal |r|={gg:.3f}  "
              f"cross |r|={qg:.3f}")
        print("   -> " + ("they SEPARATE (cross < internals): 4th axis justified"
                          if qg < min(qq, gg) else
                          "they OVERLAP (cross >= an internal): consider re-folding Growth"))

    # ---- 3. PCA scree ----
    U, S, Vt = np.linalg.svd(X / np.sqrt(len(X) - 1), full_matrices=False)
    eig = S ** 2
    ev = eig / eig.sum()
    cum = np.cumsum(ev)
    k80 = int(np.searchsorted(cum, 0.80) + 1)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    print("\n" + "=" * 60)
    print("3. SCREE  (how many factors the data actually has)")
    print("=" * 60)
    for i in range(min(10, len(ev))):
        print(f"  PC{i+1:<2} var={ev[i]:.3f}  cum={cum[i]:.3f}")
    print(f"  -> {k80} factors reach 80%, {k90} reach 90% "
          f"(expect ~4-8 if the 4 axes are roughly the structure)")

    # ---- 4. Varimax-rotated factors, axis-labelled ----
    k = max(4, k80)
    load = Vt.T[:, :k] * np.sqrt(eig[:k])          # p x k loadings
    rot = varimax(load)
    print("\n" + "=" * 60)
    print(f"4. VARIMAX FACTORS (k={k}) — top metrics per factor, [axis] labelled")
    print("=" * 60)
    for f in range(k):
        order = np.argsort(-np.abs(rot[:, f]))
        top = [(cols[i], axes[i], rot[i, f]) for i in order[:6] if abs(rot[i, f]) > 0.3]
        if not top:
            continue
        dom = pd.Series([t[1] for t in top]).mode()
        dom = dom.iloc[0] if len(dom) else "?"
        print(f"\n  Factor {f+1}  (dominant axis: {dom})")
        for name, ax, l in top:
            flag = "  <-- CROSS" if ax != dom else ""
            imp = f"  ({miss[name]:.0%} imputed)" if miss[name] > 0.25 else ""
            print(f"     {l:+.2f}  [{ax:<7}] {name}{imp}{flag}")

    print("\nDone. If (1) within>>across for all four axes and (4) factors map cleanly")
    print("to single axes, the carving matches the data — freeze the table. Cross-loaders")
    print("and any WEAK axis are the cells to revisit first.")


if __name__ == "__main__":
    main()
