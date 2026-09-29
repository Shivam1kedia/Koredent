"""
preflight.py — wiring checks. Read-only: imports, runs, asserts, prints.

Supersedes preflight_rank.py (which covered only the W2 rank change). Delete
that file once this is in place.

`ast.parse` passed on three runtime errors in W2 and would have passed on the
harness bug this sprint found. These are the assertions it cannot make.

    py preflight.py --csv universe_scored.csv

Exit code 1 on any failure, so it can gate a workflow step.

COVERAGE
  A. abstention denominators in _tier3        (rank side, shipped)
  B. forced-exit ceiling                      (sell trigger, shipped)
  C. floor + partition helpers                (13 threshold sites, shipped)
  D. comparable score-drop guard              (delta site, shipped)
  E. cross-cutting invariants
  F. drift constants
  G. ledger integrity (economics model + consumer wiring + live DB)
  H. transaction costs + the risk-free rate wiring
"""

import argparse
import ast
import inspect
import json
import sys

import numpy as np
import pandas as pd

import selector
import economics
import costs
import macro_read
import deep_metrics

FAILS = []

# Zerodha's charges are typed constants in costs.py, not a feed. This is the
# ceiling on how stale that hand-verification may get. 180 days forces a
# re-check about twice a year and cannot straddle a Union Budget (1 February),
# which is when STT and stamp duty actually move.
RATES_MAX_AGE_DAYS = 180


def skip(name, why):
    """Explicitly NOT a pass. A skipped check must never read as a green one."""
    print(f"  [SKIP] {name} — {why}")


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def section(t):
    print(f"\n{t}\n" + "-" * len(t))


# ── A. rank-side abstention ───────────────────────────────────────────────
def a_rank(df):
    section("A. abstention denominators (_tier3)")
    ok = all(hasattr(selector, n) for n in ("_rank_population", "_framework_percentile"))
    check("helpers exist", ok)
    if not ok:
        return None
    src = inspect.getsource(selector._tier3)
    check("_tier3 calls _framework_percentile", "_framework_percentile" in src)
    check("_tier3 calls _rank_population", "_rank_population" in src)
    check("old na_option='bottom' gone from _tier3",
          'na_option="bottom"' not in src and "na_option='bottom'" not in src)

    r = {}
    t1 = selector._tier1(df.copy(), 5000, r)
    ranked = selector._tier3(t1, {"philosophy": "deep_value", "demand_tilt": {}})
    lynch_out = ranked["lynch_category"].astype(str).str.strip() == "unknown"
    grn_out = ranked.get("greenblatt_sector_excluded",
                         pd.Series(False, index=ranked.index)).fillna(False).astype(bool)
    nanfrac = ranked["greenblatt_frac"].isna() & ~grn_out
    check("lynch abstainers hold NO percentile",
          bool(ranked.loc[lynch_out, "_pct_lynch"].isna().all()), f"{int(lynch_out.sum())} rows")
    check("greenblatt excluded hold NO percentile",
          bool(ranked.loc[grn_out, "_pct_greenblatt"].isna().all()), f"{int(grn_out.sum())} rows")
    check("uncomputable greenblatt hold NO percentile",
          bool(ranked.loc[nanfrac, "_pct_greenblatt"].isna().all()), f"{int(nanfrac.sum())} rows")
    for f in selector.FRAMEWORKS:
        live = int(ranked[f"_pct_{f}"].notna().sum())
        check(f"{f}: population non-empty", live > 0, f"{live} rows")
    bad = sum(1 for _, row in ranked.iterrows()
              if not set(row["_ranked_on"]) <= set(row["_applicable"]))
    check("ranked_on subset of applicable (frame)", bad == 0, f"{bad} violations")
    return ranked


# ── B. forced-exit ceiling ────────────────────────────────────────────────
def b_exit(df):
    section("B. forced-exit ceiling")
    t = {n: selector.forced_exit_threshold(n) for n in (5, 4, 3, 2, 1)}
    check("thresholds are floor(n/5)", t == {5: 1, 4: 0, 3: 0, 2: 0, 1: 0}, str(t))
    # A ceiling must never be HARSHER than the 5-denominator fraction it encodes.
    worst = max((t[n] / n) for n in (5, 4, 3, 2, 1) if t[n])
    check("no denominator exits at a harsher fraction than 1/5", worst <= 0.2 + 1e-9,
          f"max effective exit fraction {worst:.0%}")
    tick = df["ticker"].iloc[0]
    check("forced_exit_applies runs on a real row",
          isinstance(selector.forced_exit_applies(df, tick, 0), bool))
    check("score 0 exits at every denominator",
          all(selector.forced_exit_threshold(n) >= 0 for n in (5, 4, 3)))


# ── C. floor + partition helpers ──────────────────────────────────────────
def c_thresholds(df):
    section("C. floor + partition helpers")
    n, s = selector.score_applicable(df)
    check("score == score_applicable (numerator premise)",
          int((df["score"] != s).sum()) == 0,
          f"{int((df['score'] != s).sum())} mismatches")
    check("n_applicable within 1..5", bool(((n >= 1) & (n <= 5)).all()),
          ", ".join(f"n={k}:{v}" for k, v in sorted(n.value_counts().items())))

    # A FLOOR normalisation may only ADD. If it removes, the numerator premise
    # is broken — these two checks are linked, not independent.
    for k in (2, 3, 4):
        m = selector.meets_score_mask(df, k)
        removed = int(((df["score"] >= k) & ~m).sum())
        check(f"floor k={k} never removes", removed == 0,
              f"added {int((m & ~(df['score'] >= k)).sum())}, removed {removed}")

    tiers = selector.score_tiers(df)
    check("tiers only 4/3/2/None",
          set(tiers.dropna().unique()) <= {2, 3, 4}, str(sorted(set(tiers.dropna().unique()))))
    perfect = (n == s) & (n > 0)
    check("every full-marks row reaches tier 4",
          bool((tiers[perfect] == 4).all()), f"{int(perfect.sum())} full-marks rows")
    # The defect the equality partition carried independently of abstention.
    five = df["score"] == 5
    check("score==5 rows are tiered (were invisible before)",
          bool(tiers[five].notna().all()), f"{int(five.sum())} rows")


# ── D. comparable score drop ──────────────────────────────────────────────
def d_drop(df):
    section("D. comparable score-drop guard")
    th = {n: selector.score_drop_threshold(n) for n in (5, 4, 3, 2)}
    check("drop thresholds are ceil(2n/5)", th == {5: 2, 4: 2, 3: 2, 2: 1}, str(th))

    row = df.iloc[0]
    full = list(selector.FRAMEWORKS)
    # Artifact case: a framework leaves, nothing else moves -> delta 0.
    tr = {"applicable": full, "passed": ["greenblatt", "dorsey_buffett", "trajectory", "lynch"]}
    c = selector.comparable_score_drop(tr, row, 4, 3)
    check("comparable_score_drop returns the full contract",
          {"comparable", "n_common", "entry_common", "current_common", "delta", "fires"}
          <= set(c), str(sorted(c)))
    # Legacy trace with no applicable/passed must FALL BACK, never suppress.
    leg = selector.comparable_score_drop({}, row, 4, 1)
    check("legacy trace falls back to raw and still fires",
          leg["comparable"] is False and leg["fires"] is True, str(leg))
    check("headline names its basis when comparable",
          "comparable to entry" in selector.score_drop_headline("X", c, 4, 3))
    check("headline falls back cleanly when not",
          "comparable to entry" not in selector.score_drop_headline("X", leg, 4, 1))


# ── E. cross-cutting ──────────────────────────────────────────────────────
def e_wiring(df):
    section("E. cross-cutting")
    pol = {"sip_amount": 5000, "min_acceptable_score": 3, "philosophy": "deep_value",
           "demand_tilt": {}, "allocation_policy": {}, "portfolio_sizing": {"ips_target": 15}}
    res = selector.select_portfolio(df.copy(), pol, None)
    hs = res.get("holdings", [])
    check("selection produces holdings", len(hs) > 0, str(len(hs)))
    if not hs:
        return
    check("every trace carries ranked_on", all("ranked_on" in h["_trace"] for h in hs))
    check("ranked_on subset of applicable (trace)",
          all(set(h["_trace"]["ranked_on"]) <= set(h["_trace"]["applicable"]) for h in hs))
    try:
        json.dumps([h["_trace"] for h in hs])
        check("trace is JSON-serialisable", True)
    except (TypeError, ValueError) as e:
        check("trace is JSON-serialisable", False, str(e))

    # No consumer should be gating on the raw integer any more.
    # A check that cannot read its evidence must FAIL, not pass. The first
    # version of this swallowed OSError and reported PASS on zero files read.
    import re
    leftover, read = 0, []
    for f in ("app.py", "portfolio_tracker.py"):
        try:
            leftover += len(re.findall(r"\[.score.\]\s*(>=|==)\s*[0-9]",
                                       open(f, encoding="utf-8").read()))
            read.append(f)
        except OSError:
            pass
    check("no raw score thresholds left in consumers",
          len(read) == 2 and leftover == 0,
          f"{leftover} found in {read or 'NO FILES READ — run from the repo root'}")

    div = [h["ticker"] for h in hs
           if set(h["_trace"]["ranked_on"]) != set(h["_trace"]["applicable"])]
    print(f"\n  note: {len(div)}/{len(hs)} holdings rank on fewer frameworks than apply"
          + (f": {', '.join(div)}" if div else ""))
    print("        (the gate still counts `applicable` — that divergence is C1's)")


# ── F. drift constants ────────────────────────────────────────────────────
def f_drift():
    section("F. drift constants (two denominators)")
    import re
    from pathlib import Path

    # The old shared constant must be gone everywhere. A check that cannot read
    # its evidence must FAIL, not pass — so a zero-file walk is a failure, not
    # a silent PASS on nothing.
    files = sorted(Path(".").glob("*.py"))
    hits = []
    # Split so this file's own source does not match the pattern it searches
    # for. Written whole, the check reported preflight.py as a live use of the
    # dead constant -- a grep that cannot tell "still in use" from "mentioned
    # by the checker" is a false positive generator.
    needle = "DRIFT_MATERIAL" + "_CONTINUOUS"
    for p in files:
        try:
            if needle in p.read_text(encoding="utf-8"):
                hits.append(p.name)
        except OSError:
            pass
    check("old shared constant is gone",
          len(files) > 0 and not hits,
          f"{len(files)} files read"
          + (f", still present in {hits}" if hits else "")
          + ("" if files else " — run from the repo root"))

    # Assert the RELATION, not the value 0.25. When 3b's per-framework
    # measurement moves the floor, this check keeps holding without an edit —
    # a check pinned to the literal would have to be edited in lockstep, which
    # is how the two constants drift apart again.
    check("total bound is derived from the framework floor",
          abs(selector.DRIFT_MATERIAL_TOTAL
              - len(selector.FRAMEWORKS) * selector.DRIFT_FLOOR_FRAMEWORK) < 1e-9,
          f"{selector.DRIFT_MATERIAL_TOTAL} vs "
          f"{len(selector.FRAMEWORKS)}x{selector.DRIFT_FLOOR_FRAMEWORK}")

    # Behavioural, not textual: the two sites must read DIFFERENT constants.
    base = {f: 0.5 for f in selector.FRAMEWORKS}
    mk = lambda fr, sc: {"fracs": fr, "score_continuous": sc}

    # (i) five sub-floor moves. Total 0.20 < 0.25 -> must NOT reach the alert,
    #     and crowns no mover. This was the genuine orphan: "your score moved,
    #     cause unattributable", with unattributed = 0.
    wob = selector._continuous_drift(
        mk(base, 2.5), mk({k: v + 0.04 for k, v in base.items()}, 2.7))
    check("five sub-floor moves stay below the total bound",
          abs(wob["delta"]) < selector.DRIFT_MATERIAL_TOTAL
          and wob["largest_move"] is None
          and abs(wob["unattributed"]) < 1e-9,
          f"delta={wob['delta']} largest={wob['largest_move']}")

    # (ii) one clearly material move -> fires AND names its mover.
    big = dict(base); big["graham"] = 0.5 + 0.30
    mv = selector._continuous_drift(mk(base, 2.5), mk(big, 2.8))
    check("a material single move fires and names its mover",
          abs(mv["delta"]) >= selector.DRIFT_MATERIAL_TOTAL
          and mv["largest_move"] == "graham",
          f"delta={mv['delta']} largest={mv['largest_move']}")

    # (iii) a framework becomes scoreable. largest_move is None here and that
    #     is CORRECT, not an orphan — the cause is named via `unattributed`,
    #     which app.py renders in the same sentence as the number. Asserting
    #     this pins the distinction so a later "fix" cannot collapse the two.
    ent = dict(base); ent[selector.FRAMEWORKS[0]] = None
    became = selector._continuous_drift(mk(ent, 2.0), mk(base, 2.5))
    check("became-scoreable reports unattributed, not a false mover",
          became["largest_move"] is None
          and abs(became["unattributed"]) >= selector.DRIFT_FLOOR_FRAMEWORK,
          f"unattributed={became['unattributed']} "
          f"unmeasured={became['unmeasured']}")

# ── G. ledger integrity ───────────────────────────────────────────────────
def _txn(i, d, tk, sh, px, tt, bpx=100.0, amt=None, cost=None):
    """cost=None means the row carries NO cost_inr — a row written before cost
    tracking existed. That is the legacy shape, and the checks below use it to
    pin that legacy ledgers still replay to exactly their old numbers."""
    return {"id": str(i), "created_at": "2026-01-01T00:00:%02d" % i,
            "transaction_date": d, "ticker": tk, "shares": sh, "price": px,
            "amount_inr": round(sh * px, 2) if amt is None else amt,
            "transaction_type": tt, "nifty_price": bpx, "cost_inr": cost}


def g1_model():
    """The economics model, asserted on synthetic ledgers. No DB, no CSV.

    These are the properties the money layer must never lose, each pinned to the
    concrete failure it prevents. They are cheap; the bugs they catch were not.
    """
    section("G1. economics model invariants")

    # The case that kills the naive design: netting withdrawals off the
    # denominator gives 10,000 - 10,000 = 0 and a return of infinity.
    dbl = [_txn(1, "2026-01-01", "A.NS", 100, 100, "buy"),
           _txn(2, "2026-03-01", "A.NS", 50, 200, "sell"),
           _txn(3, "2026-03-02", "CASH", 0, 0, "withdrawal", amt=10000.0)]
    e = economics.portfolio_economics(dbl, 10000)
    check("withdrawal does not reduce external capital",
          e["external_capital"] == 10000.0 and e["return_pct"] == 100.0,
          f"ext={e['external_capital']} ret={e['return_pct']}")

    # Realising a loss must not RAISE reported return. This is the original bug:
    # invested = surviving cost basis deletes cost and outcome together.
    loss = [_txn(1, "2026-01-01", "A.NS", 100, 100, "buy"),
            _txn(2, "2026-03-01", "A.NS", 100, 50, "sell")]
    el = economics.portfolio_economics(loss, 0)
    check("realising a loss lowers return, never raises it",
          el["return_pct"] is not None and el["return_pct"] < 0,
          f"ret={el['return_pct']} realized={el['realized_pnl']}")

    # A rotation is not new capital and is not a benchmark event.
    rot = [_txn(1, "2026-01-01", "A.NS", 100, 100, "buy", bpx=250.0),
           _txn(2, "2026-03-01", "A.NS", 100, 100, "sell", bpx=300.0),
           _txn(3, "2026-03-01", "B.NS", 50, 200, "buy", bpx=300.0)]
    er = economics.portfolio_economics(rot, 10000, 300.0)
    check("rotation adds no external capital", er["external_capital"] == 10000.0,
          str(er["external_capital"]))
    check("rotation adds no shadow units", abs(er["shadow_units"] - 40.0) < 1e-9,
          f"{er['shadow_units']:.4f}")

    # Sell-and-hold must not move the shadow. Summing the stored nifty_units
    # column did, which made an idle sale look like a withdrawal.
    hold = rot[:2]
    eh = economics.portfolio_economics(hold, 0, 300.0)
    check("sell-and-hold leaves the shadow untouched",
          abs(eh["shadow_units"] - 40.0) < 1e-9, f"{eh['shadow_units']:.4f}")

    # Cash is a floor, and an impossible withdrawal is surfaced not swallowed.
    over = [_txn(1, "2026-01-01", "A.NS", 100, 100, "buy"),
            _txn(2, "2026-03-01", "CASH", 0, 0, "withdrawal", amt=5000.0)]
    eo = economics.portfolio_economics(over, 10000)
    check("over-withdrawal is clamped AND flagged",
          eo["cash_balance"] >= 0 and eo["unreconciled_withdrawal"] == 5000.0,
          f"cash={eo['cash_balance']} unrec={eo['unreconciled_withdrawal']}")

    # ── Sprint 16: the same ledgers, COSTED ──────────────────────────────
    # A cost is money that left, so it must reduce cash and raise external
    # capital on a buy that cash cannot cover. If costs were a display-time
    # adjustment instead, every one of these would still pass with costs
    # silently absent from the money — which is the failure being pinned.
    c_buy = costs.buy_cost(10000)
    c_sell = costs.sell_cost(10000)
    cdbl = [_txn(1, "2026-01-01", "A.NS", 100, 100, "buy", cost=c_buy),
            _txn(2, "2026-03-01", "A.NS", 50, 200, "sell", cost=c_sell),
            _txn(3, "2026-03-02", "CASH", 0, 0, "withdrawal", amt=10000.0, cost=0.0)]
    ec = economics.portfolio_economics(cdbl, 10000)
    check("costs raise external capital on an uncovered buy",
          abs(ec["external_capital"] - (10000.0 + c_buy)) < 0.02,
          f"ext={ec['external_capital']} vs 10000+{c_buy}")
    check("total_costs_paid is the sum of the row costs",
          abs(ec["total_costs_paid"] - (c_buy + c_sell)) < 0.02,
          f"{ec['total_costs_paid']} vs {c_buy + c_sell}")
    # The split is what the UI shows. Buying is ~0.12% at any size; selling
    # carries the flat DP charge. A blended total hides the asymmetry that
    # every exit rule has to be designed around.
    check("buy and sell charges split and still sum to the total",
          abs(ec["buy_costs_paid"] - c_buy) < 0.02
          and abs(ec["sell_costs_paid"] - c_sell) < 0.02
          and abs(ec["buy_costs_paid"] + ec["sell_costs_paid"]
                  - ec["total_costs_paid"]) < 0.02,
          f"buy={ec['buy_costs_paid']} sell={ec['sell_costs_paid']}")
    # Gross must be EXACT, not a counterfactual: gross - charges == net.
    check("gross P&L minus charges equals net P&L",
          abs(ec["gross_pnl"] - ec["total_costs_paid"] - ec["total_pnl"]) < 0.02,
          f"{ec['gross_pnl']} - {ec['total_costs_paid']} vs {ec['total_pnl']}")
    check("gross return is never below net return",
          ec["gross_return_pct"] >= ec["return_pct"],
          f"gross {ec['gross_return_pct']}% vs net {ec['return_pct']}%")
    check("costs lower return, never raise it",
          ec["return_pct"] < e["return_pct"],
          f"costed {ec['return_pct']}% vs gross {e['return_pct']}%")

    # NULL cost is UNKNOWN, not zero. `False` carrying two meanings is the bug
    # score_history.applicable exists to prevent; this is the same shape.
    mixed = [_txn(1, "2026-01-01", "A.NS", 100, 100, "buy"),
             _txn(2, "2026-02-01", "B.NS", 100, 100, "buy", cost=c_buy)]
    em = economics.portfolio_economics(mixed, 20000)
    check("a row with no cost_inr is COUNTED, not silently zero",
          em["cost_rows_missing"] == 1, f"{em['cost_rows_missing']} missing")

    # A legacy ledger must replay to EXACTLY its pre-Sprint-16 numbers. If this
    # fails, the cost change rewrote history rather than extending it.
    check("legacy (uncosted) ledger is unchanged by the cost term",
          e["external_capital"] == 10000.0 and e["return_pct"] == 100.0
          and e["total_costs_paid"] == 0.0,
          f"ext={e['external_capital']} ret={e['return_pct']} "
          f"costs={e['total_costs_paid']}")

    # The shadow buys SECURITIES, not friction. A rotation draws a little
    # external capital purely to cover its own costs; crediting the benchmark
    # with units for that money makes the benchmark grow with churn.
    crot = [_txn(1, "2026-01-01", "A.NS", 100, 100, "buy", bpx=250.0, cost=c_buy),
            _txn(2, "2026-03-01", "A.NS", 100, 100, "sell", bpx=300.0, cost=c_sell),
            _txn(3, "2026-03-01", "B.NS", 50, 200, "buy", bpx=300.0, cost=c_buy)]
    ecr = economics.portfolio_economics(crot, 10000, 300.0)
    check("a costed rotation still adds NO shadow units",
          abs(ecr["shadow_units"] - 40.0) < 1e-6, f"{ecr['shadow_units']:.6f}")
    check("a costed rotation draws external capital only for its costs",
          abs(ecr["external_capital"] - (10000.0 + 2 * c_buy + c_sell)) < 0.02,
          f"{ecr['external_capital']}")

    # The decomposition identity, on every case above. THREE terms since
    # Sprint 16 — realized and unrealized both stay GROSS and costs are their
    # own line, so unrealized keeps meaning market_value minus surviving cost
    # basis and stays checkable against holdings.price_at_entry.
    for nm, ecc in (("doubling", e), ("loss", el), ("rotation", er),
                    ("hold", eh), ("over", eo), ("costed", ec),
                    ("costed rotation", ecr), ("mixed", em)):
        check(f"identity holds ({nm}): realized + unrealized - costs == total",
              abs(ecc["unrealized_pnl"] + ecc["realized_pnl"]
                  - ecc["total_costs_paid"] - ecc["total_pnl"]) < 0.02,
              f"{ecc['realized_pnl']} + {ecc['unrealized_pnl']} - "
              f"{ecc['total_costs_paid']} vs {ecc['total_pnl']}")
        check(f"cash never goes negative ({nm})", ecc["cash_balance"] >= 0,
              f"{ecc['cash_balance']}")

    # No capital, no percentage. A wrong number is worse than no number.
    check("return_pct is None with no external capital",
          economics.portfolio_economics([], 0)["return_pct"] is None)

    # XIRR must see external flows only, never gross trades.
    d, a = economics.xirr_flows(economics.portfolio_economics(rot, 10000))
    check("xirr_flows carries external flows only, not every trade",
          d is not None and len(d) == 2, f"{len(d) if d else 0} points for a 3-row ledger")


def _fn_args(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return [a.arg for a in n.args.args]
    return None


def g2_wiring():
    """Structural, via AST — app.py cannot be imported (Streamlit runs on import).

    Signatures are the regression guard: reverting the money layer changes them,
    and a changed signature fails here before anything reaches a user.
    """
    section("G2. consumer wiring")
    trees, unread = {}, []
    for f in ("app.py", "portfolio_tracker.py"):
        try:
            trees[f] = ast.parse(open(f, encoding="utf-8").read())
        except (OSError, SyntaxError):
            unread.append(f)
    if unread:
        check("consumers are readable", False,
              f"could not parse {unread} — run from the repo root")
        return

    for f in trees:
        names = {n.names[0].name if isinstance(n, ast.Import) else n.module
                 for n in ast.walk(trees[f]) if isinstance(n, (ast.Import, ast.ImportFrom))}
        check(f"{f} imports economics", "economics" in names)

    a = trees["app.py"]
    for fn in ("portfolio_money", "load_txns", "record_withdrawal", "live_price"):
        check(f"app.py defines {fn}", _fn_args(a, fn) is not None)

    # The old signature took the client and re-queried; the new one takes the
    # already-computed economics. Reverting is what this catches.
    args = _fn_args(a, "compute_portfolio_xirr")
    check("app.compute_portfolio_xirr takes econ, not a db client",
          args is not None and args[0] == "econ", str(args))
    targs = _fn_args(trees["portfolio_tracker.py"], "compute_xirr_standalone")
    check("tracker.compute_xirr_standalone takes econ, not a db client",
          targs is not None and targs[0] == "econ", str(targs))
    pargs = _fn_args(a, "portfolio_money")
    check("portfolio_money accepts a benchmark ticker",
          pargs is not None and "benchmark_ticker" in pargs, str(pargs))

    # Sprint 16. The ledger is the only place a cost can be recorded, so BOTH
    # halves have to be wired: load_txns must select the column, and
    # record_transaction must write it. Either one missing and costs silently
    # read as zero everywhere — which is indistinguishable from the state this
    # sprint replaced.
    check("app.py imports costs", "costs" in {
        (n.names[0].name if isinstance(n, ast.Import) else n.module)
        for n in ast.walk(a) if isinstance(n, (ast.Import, ast.ImportFrom))})
    lt = next((n for n in ast.walk(a)
               if isinstance(n, ast.FunctionDef) and n.name == "load_txns"), None)
    check("load_txns selects cost_inr",
          lt is not None and "cost_inr" in ast.unparse(lt))
    rt = next((n for n in ast.walk(a)
               if isinstance(n, ast.FunctionDef) and n.name == "record_transaction"), None)
    rt_src = ast.unparse(rt) if rt is not None else ""
    check("record_transaction writes cost_inr", "cost_inr" in rt_src)
    check("record_transaction prices the sell side separately",
          "sell_cost" in rt_src and "buy_cost" in rt_src)
    rw = next((n for n in ast.walk(a)
               if isinstance(n, ast.FunctionDef) and n.name == "record_withdrawal"), None)
    check("record_withdrawal writes a KNOWN zero cost, not NULL",
          rw is not None and "cost_inr" in ast.unparse(rw))

    # The tracker must store which rate it used. rfr_used and its PDF stamp
    # both already existed and nothing ever wrote the value, so the stamp had
    # never rendered once — a live rate makes that provenance mandatory.
    tsrc = open("portfolio_tracker.py", encoding="utf-8").read()
    check("tracker stores rfr_used and rfr_status",
          '"rfr_used", "rfr_status"' in tsrc)

    # The dead denominator. Any of these forms is the Sprint-14 bug returning.
    # Plus the Sprint-15 asymmetry: crediting the portfolio's CASH to the
    # benchmark. The shadow tracks external flows, so it never sold when you
    # did; adding cash counts the same proceeds on both sides. Only WITHDRAWN
    # belongs there, mirroring the portfolio numerator exactly.
    import re
    dead = re.compile(r"(_cv\s*-\s*_inv|current_total_value\s*-\s*total_invested"
                      r"|current_val\s*-\s*total_invested"
                      r"|last_shadow\s*\+\s*_econ\[.cash_balance.\]"
                      r"|nifty_shadow_value.{0,4}\+\s*hist_df\[.cash_balance.\])")
    hits = []
    for f in trees:
        for i, ln in enumerate(open(f, encoding="utf-8").read().split("\n"), 1):
            if dead.search(ln):
                hits.append(f"{f}:{i}")
    check("surviving-cost-basis denominator is gone", not hits, str(hits))


def g3_db(required=False):
    """Live reconciliation. Every rupee in holdings must have a ledger row.

    This is the check that protects FUTURE features: any code path that mutates
    holdings without writing the ledger breaks it, whatever the feature is.
    """
    section("G3. live ledger reconciliation")
    import os
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY")
    if not (url and key):
        if required:
            check("database is reachable", False,
                  "SUPABASE_URL/SUPABASE_KEY not set but --db was requested")
        else:
            skip("live ledger reconciliation",
                 "SUPABASE_URL/SUPABASE_KEY not set; pass --db to require it")
        return
    try:
        from supabase import create_client
        sb = create_client(url, key)
        txns = sb.table("sip_transactions").select(
            "id, created_at, portfolio_id, ticker, shares, price, amount_inr, "
            "transaction_type, transaction_date, nifty_price").execute().data or []
        holds = sb.table("holdings").select("portfolio_id, ticker, shares").execute().data or []
    except Exception as ex:
        check("database is reachable", False, f"{type(ex).__name__}: {ex}")
        return

    net = {}
    for t in txns:
        k = (t["portfolio_id"], t.get("ticker"))
        sh = float(t.get("shares") or 0)
        tt = str(t.get("transaction_type") or "buy").lower()
        if tt == "withdrawal":
            continue
        net[k] = net.get(k, 0.0) + (sh if tt == "buy" else -sh)
    held = {}
    for h in holds:
        k = (h["portfolio_id"], h.get("ticker"))
        held[k] = held.get(k, 0.0) + float(h.get("shares") or 0)

    drift = [f"{k[0]}/{k[1]}: ledger {net.get(k, 0)} vs held {held.get(k, 0)}"
             for k in set(net) | set(held)
             if abs(net.get(k, 0.0) - held.get(k, 0.0)) > 1e-6]
    check("ledger shares reconcile to holdings shares", not drift,
          "; ".join(drift[:5]) + (f" (+{len(drift)-5} more)" if len(drift) > 5 else ""))

    junk = [t["id"] for t in txns if float(t.get("amount_inr") or 0) <= 0]
    check("no zero-amount ledger rows", not junk, f"{len(junk)} rows")

    bad_type = sorted({str(t.get("transaction_type")) for t in txns}
                      - {"buy", "sell", "withdrawal"})
    check("transaction_type values are all known", not bad_type, str(bad_type))

    by_port = {}
    for t in txns:
        by_port.setdefault(t["portfolio_id"], []).append(t)
    unrec, incomplete = [], []
    for pid, rows in by_port.items():
        led = economics.replay_ledger(rows)
        if led["unreconciled_withdrawal"] > 0.005:
            unrec.append(f"{pid}:{led['unreconciled_withdrawal']:.2f}")
        if led["shadow_incomplete"]:
            incomplete.append(str(pid))
    check("no unreconciled withdrawals", not unrec, str(unrec))
    check("every contribution carries a benchmark price", not incomplete,
          f"shadow understates for portfolios {incomplete}" if incomplete else "")


# ── H. transaction costs + risk-free rate wiring ──────────────────────────
def h_costs_and_rates():
    """The Sprint 16 foundation. Costs must be computable and RIGHT, and there
    must be exactly ONE risk-free rate for portfolio metrics."""
    section("H1. transaction cost model")

    # Purity. costs.py is the single source of truth for rates and is imported
    # by economics consumers, the backtest and the UI; an import here is how it
    # acquires a network or pandas dependency by accident.
    tree = ast.parse(open("costs.py", encoding="utf-8").read())
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    check("costs.py imports nothing at all", not imports,
          str([getattr(n, "module", None) or n.names[0].name for n in imports]))

    # The two anchors from the sprint brief, recomputed rather than restated.
    # If Zerodha moves a rate and RATES is updated, these MOVE — and they
    # should, because a check pinned to a stale number is a check that lies.
    rt333 = costs.round_trip_pct(333) * 100
    rt50k = costs.round_trip_pct(50000) * 100
    check("round_trip_pct(333) is ~4.83%", abs(rt333 - 4.83) < 0.02, f"{rt333:.3f}%")
    check("round_trip_pct(50000) is ~0.25%", abs(rt50k - 0.25) < 0.02, f"{rt50k:.3f}%")

    # The flat DP charge is the whole story at retail sizes. This is the
    # property every exit rule in Sprint 20 will be designed against.
    dp = costs.RATES["dp_per_scrip_sell"]
    share = dp / costs.round_trip_cost(333)
    check("the flat DP charge dominates a small position", share > 0.9,
          f"{share:.0%} of a Rs 333 round trip")
    check("selling costs far more than buying at retail size",
          costs.sell_cost(333) > 20 * costs.buy_cost(333),
          f"sell {costs.sell_cost(333)} vs buy {costs.buy_cost(333)}")

    # Break-evens must invert the cost function exactly, not approximately.
    for target in (0.010, 0.0075, 0.005):
        v = costs.min_position_for_cost_pct(target)
        check(f"break-even inverts cleanly at {target*100:.2f}%",
              v is not None and abs(costs.round_trip_pct(v) - target) < 1e-6,
              f"Rs {v:,.2f}" if v else "None")
    check("an unreachable cost target returns None, not a number",
          costs.min_position_for_cost_pct(0.001) is None)

    # BSE transaction charge is 22% higher, and the universe carries both.
    check("exchange is derived from the ticker suffix",
          costs.exchange_for("X.BO") == "BSE" and costs.exchange_for("X.NS") == "NSE"
          and costs.exchange_for(None) == "NSE")
    check("BSE costs more than NSE",
          costs.round_trip_cost(5000, "BSE") > costs.round_trip_cost(5000, "NSE"),
          f"{costs.round_trip_cost(5000,'BSE')} vs {costs.round_trip_cost(5000,'NSE')}")

    # A bad input must not become a free trade.
    check("non-positive or unparseable values yield no cost and no percentage",
          costs.buy_cost(0) == 0.0 and costs.buy_cost(None) == 0.0
          and costs.round_trip_pct(0) is None
          and costs.net_return(None, 0.1) is None)

    # Costs must make a flat round trip a LOSS. If this ever passes at zero,
    # the model has been disconnected.
    nr = costs.net_return(500, 0.0)
    check("a flat round trip on a Rs 500 position is a loss",
          nr is not None and nr < -0.03, f"{nr*100:.2f}%")

    # THE RATES ARE NOT FETCHED. They are typed constants with a verification
    # date, because no broker publishes them as a feed and they move on a
    # Budget, not a ticker. That is the right design — and it has exactly one
    # failure mode: nobody re-checks them, and the whole cost model silently
    # describes last year's India.
    #
    # STT moved in a Union Budget before and will again. The Budget is 1
    # February, so a 180-day ceiling forces a re-check roughly twice a year and
    # cannot skip a Budget. This FAILS rather than warns: a control that only
    # prints is the control this sprint was written to replace.
    import datetime as _dt
    try:
        _verified = _dt.date.fromisoformat(costs.RATES_VERIFIED)
        _age = (_dt.date.today() - _verified).days
    except (ValueError, TypeError):
        _verified, _age = None, None
    check("the cost rates carry a parseable verification date", _verified is not None,
          str(costs.RATES_VERIFIED))
    if _age is not None:
        check(f"cost rates re-verified within {RATES_MAX_AGE_DAYS} days",
              _age <= RATES_MAX_AGE_DAYS,
              f"verified {costs.RATES_VERIFIED}, {_age} days ago — re-check "
              f"https://zerodha.com/charges/ and update RATES_VERIFIED in costs.py")

    print(f"\n  note: rates are typed constants verified {costs.RATES_VERIFIED}, "
          f"not a live feed. No broker publishes them as one.")
    print(f"        Rs 5,000 position round trip: "
          f"{costs.round_trip_pct(5000)*100:.2f}%; Rs 500: "
          f"{costs.round_trip_pct(500)*100:.2f}%")

    section("H2. risk-free rate — one rate, one place")

    import portfolio_tracker as pt

    # The live path must actually be reached. A constant returned directly is
    # the regression this catches: the function keeps its name and its
    # signature, and quietly stops reading the series.
    src = inspect.getsource(pt.get_india_rfr_status)
    check("get_india_rfr_status reads macro_read", "macro_read" in src)
    check("get_india_rfr delegates rather than returning a constant",
          "get_india_rfr_status" in inspect.getsource(pt.get_india_rfr))

    # EXACTLY ONE fallback constant. A second 0.07 in the tracker would be the
    # original duplication bug restored: one rate in two files, invisible in
    # both, free to drift apart.
    tsrc_rfr = open("portfolio_tracker.py", encoding="utf-8").read()
    check("the tracker defines no rate constant of its own",
          "INDIA_RFR_FALLBACK = " not in tsrc_rfr and "INDIA_RFR = " not in tsrc_rfr)

    rate, status = pt.get_india_rfr_status()
    check("the rate is a plausible decimal, not a percentage",
          isinstance(rate, float) and 0.0 < rate < 0.20, f"{rate}")
    if status == "ok":
        check("the live series answered", True, f"{rate*100:.3f}%")
    else:
        # NOT a pass and NOT a silent fallback. The rate still works; the fact
        # that it came from the fallback is what has to stay visible.
        skip("live risk-free rate", f"status={status}, using macro_read "
                                    f"fallback {macro_read.FALLBACK_INDIA_RFR*100:.2f}%")

    # macro_series.json must have a real consumer. That was the whole point:
    # seven weeks of daily Tavily credits with nothing reading the output.
    check("macro_series.json is readable and non-empty",
          len(macro_read.read_series()) > 0, f"{len(macro_read.read_series())} readings")

    # ONE definition of operative_value, shared by writer and reader.
    import macro_fetch
    check("macro_fetch shares the reader's operative_value",
          macro_fetch.operative_value is macro_read.operative_value)

    # A non-ok status must never hand back a number.
    for bogus in ("nonexistent_field",):
        v, st = macro_read.operative(bogus)
        check(f"an unknown field yields no value ({st})", v is None, str(v))

    section("H3. scoring constant is FROZEN, and monitored")

    # The scorer must NOT be on the live rate. A stock's Graham score moving
    # because the G-Sec moved 4bp is noise entering a stored score, and it
    # breaks comparability with every archived snapshot.
    # STRUCTURAL, via AST — not a text grep. deep_metrics DOCUMENTS the live
    # reader in the comment block above the constant, and a grep for the name
    # matches that documentation and fails on it. Section F hit the same class
    # of false positive; a check that cannot tell "imported" from "mentioned"
    # is a false-positive generator, and the fix is to read imports, not text.
    dm_tree = ast.parse(open("deep_metrics.py", encoding="utf-8").read())
    dm_imports = set()
    for n in ast.walk(dm_tree):
        if isinstance(n, ast.Import):
            dm_imports |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            dm_imports.add(n.module.split(".")[0])
    check("deep_metrics does not import the live rate reader",
          not ({"macro_read", "macro_fetch"} & dm_imports),
          str(sorted({"macro_read", "macro_fetch"} & dm_imports)))
    check("the scoring constant is dated",
          isinstance(getattr(deep_metrics, "INDIA_10Y_BOND_RATE_AS_OF", None), str),
          str(getattr(deep_metrics, "INDIA_10Y_BOND_RATE_AS_OF", None)))

    mon = macro_read.rate_monitor(deep_metrics.INDIA_10Y_BOND_RATE)
    if mon["status"] != "ok":
        skip("scoring-constant drift", f"live rate unavailable ({mon['status']})")
    else:
        # FAILS past the pre-registered band. A drift this large means the
        # frozen constant no longer describes the rate environment the Graham
        # spread assumes, and somebody has to DECIDE — re-score with a
        # SCHEMA_VERSION bump, or move the constant and its as-of date. A red
        # build is the mechanism that forces the decision; this control runs
        # after the universe commit, so failing it costs no data.
        check(f"frozen {mon['frozen_pct']}% is within "
              f"{macro_read.REEXAMINE_BP:.0f}bp of live {mon['live_pct']}%",
              not mon["exceeds"], f"drift {mon['drift_bp']:+.1f}bp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="universe_scored.csv")
    ap.add_argument("--db", action="store_true",
                    help="require the live ledger reconciliation (G3) to run")
    args = ap.parse_args()
    df = pd.read_csv(args.csv, low_memory=False)
    print(f"preflight — {len(df)} rows")
    a_rank(df)
    b_exit(df)
    c_thresholds(df)
    d_drop(df)
    e_wiring(df)
    f_drift()
    g1_model()
    g2_wiring()
    g3_db(required=args.db)
    h_costs_and_rates()
    print("\n" + ("ALL CHECKS PASSED" if not FAILS
                  else f"{len(FAILS)} FAILED: " + "; ".join(FAILS)))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
