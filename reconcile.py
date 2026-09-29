"""
reconcile.py — proves the archive is re-scoreable at the TERMINAL scoring layer.
Feeds each frozen universe_scored.csv row back through the terminal scorers (in
the real two-phase order: per-row -> universe-wide Greenblatt ranks -> re-run
verdicts) and checks they reproduce the stored values.

Coverage is AUTOMATIC: every column any terminal scorer writes is discovered by
introspection, so the check can't miss a column someone forgot to list.

Usage:  py reconcile.py [universe_scored.csv]
"""
import sys, math, re, inspect
import pandas as pd, numpy as np

# On the dev box the real yfinance is importable; the stub line is sandbox-only.
try:
    import deep_metrics as dm
except ModuleNotFoundError:
    sys.path.insert(0, "_stubs"); import deep_metrics as dm

CSV = sys.argv[1] if len(sys.argv) > 1 else "universe_scored.csv"

TERMINALS = ["compute_classification", "compute_spectrum_scores",
             "compute_trajectory_score", "compute_quality_gate",
             "compute_framework_verdicts", "compute_axis_scores"]

def discover_outputs():
    """Every column the terminal scorers ASSIGN — the full set to verify."""
    outs = set()
    for fn in TERMINALS:
        src = inspect.getsource(getattr(dm, fn))
        outs |= set(re.findall(r'data\[[\'"]([\w]+)[\'"]\]\s*=', src))
    return sorted(outs)

def eqv(a, b, tol=1e-6):
    an = a is None or (isinstance(a, float) and math.isnan(a))
    bn = b is None or (isinstance(b, float) and math.isnan(b))
    if an and bn: return True
    if an != bn: return False
    if isinstance(a,(int,float,np.integer,np.floating)) and isinstance(b,(int,float,np.integer,np.floating)):
        try: return abs(float(a)-float(b)) < tol
        except: return a == b
    return str(a) == str(b)

def rescore_all(datas):
    for d in datas:
        dm.compute_classification(d); dm.compute_spectrum_scores(d)
        dm.compute_trajectory_score(d); dm.compute_quality_gate(d)
        dm.compute_axis_scores(d)
        dm.compute_framework_verdicts(d)        # provisional
    dm.compute_greenblatt_ranks(datas)          # universe-wide rank pass
    for d in datas:
        dm.compute_trajectory_score(d); dm.compute_framework_verdicts(d)  # final

# Terminal scorers changed semantics at v4 (W2 archetype engine): lynch_category
# is no longer produced by the old eps_cv ladder, so v4 code re-scoring a v3 row
# reproduces a DIFFERENT category — and every column downstream of it. That is
# the declared break, not archive corruption. Reconciling across it would report
# ~3,100 "mismatches" that are the intended result, which is how a guard becomes
# noise and stops being read.
MIN_RECONCILABLE_SCHEMA = 4


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows x {df.shape[1]} cols from {CSV}\n")

    _sv = None
    if "schema_version" in df.columns and len(df):
        try:
            _sv = int(pd.to_numeric(df["schema_version"], errors="coerce").dropna().iloc[0])
        except Exception:
            _sv = None
    if _sv is not None and _sv < MIN_RECONCILABLE_SCHEMA:
        print(f"SKIP — schema_version={_sv}, below the v{MIN_RECONCILABLE_SCHEMA} "
              f"floor. Pre-v4 rows were scored by the old Lynch-category ladder; "
              f"re-scoring them with current code is EXPECTED to differ and proves "
              f"nothing about archive integrity.\n"
              f"Reconcile is valid WITHIN a schema_version, never across one.")
        sys.exit(0)
    if _sv is None:
        print("WARNING: no schema_version column — cannot confirm this archive is "
              "v4. Results below may reflect the v3->v4 break rather than drift.\n")

    outputs = discover_outputs()
    present = [c for c in outputs if c in df.columns]
    absent  = [c for c in outputs if c not in df.columns]
    print(f"Terminal-scorer output columns discovered: {len(outputs)}")
    if absent:
        print(f"  NOT in CSV (can't verify — possible archive gap): {absent}")
    print(f"  verifying {len(present)} present columns\n")

    datas, stored = [], []
    for _, row in df.iterrows():
        d = {k: (None if (isinstance(v,float) and math.isnan(v)) else v) for k,v in row.items()}
        stored.append({c: d.get(c) for c in present})
        datas.append(d)

    rescore_all(datas)

    mismatch = {c: 0 for c in present}; samples = {c: [] for c in present}
    for i, d in enumerate(datas):
        for c in present:
            if not eqv(stored[i][c], d.get(c)):
                mismatch[c] += 1
                if len(samples[c]) < 4:
                    samples[c].append((df.iloc[i]["ticker"], stored[i][c], d.get(c)))

    bad = {c: n for c, n in mismatch.items() if n}
    print(f"{'column':<34}{'mismatches':>10}"); print("-"*44)
    for c in present:
        print(f"{c:<34}{mismatch[c]:>10}{'  <--' if mismatch[c] else ''}")
    print("-"*44)
    print(f"TOTAL mismatches: {sum(mismatch.values())} | columns with drift: {len(bad)}\n")
    for c in bad:
        print(f"[{c}]:"); [print(f"    {t}: {s!r} -> {r!r}") for t,s,r in samples[c]]

    ok = (not bad) and (not absent)
    print("\n" + ("RECONCILE PASS — archive is re-scoreable at the terminal layer."
                  if ok else "RECONCILE FAIL — see columns marked <-- and any 'NOT in CSV' above."))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
