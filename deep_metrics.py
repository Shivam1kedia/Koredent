"""
DEEP METRICS MODULE
===================
Computes all 90 new columns from the 7-book extraction (Sprint 6).
Called by universe_updater.py after basic fundamentals are fetched.

Books: Graham, Greenblatt, Dorsey, Lynch, Buffett, Schilit, Mulford
Categories: Balance Sheet, Earnings Quality, Valuation, Growth, Moat,
            Dividend, Management, Manipulation Flags, Classification
"""

import math
import pandas as pd
from datetime import datetime, timezone

# W2 archetype engine. HARD dependency, unlike the SOFT axis_scores import
# further down: if this is missing, lynch_category cannot be derived and every
# Lynch score downstream is silently wrong. Fail at import, not at row 3,000.
from archetype import assign_archetype, lynch_category_from

# ─── Constants ───
# INDIA_10Y_BOND_RATE — a FROZEN SCORING CONSTANT. Not a stale number waiting to
# be made live; a parameter the entire score archive was computed against.
#
# Frozen at 7.0, as-of 2026-07-01. Sprint 16 deliberately left it frozen while
# making the PORTFOLIO risk-free rate live (portfolio_tracker.get_india_rfr).
# The two are different objects and the distinction is the whole point:
#
#   - A Sharpe ratio is recomputed from scratch every day. Feeding it today's
#     G-Sec is simply correct, and changes nothing that was stored yesterday.
#   - This constant feeds graham_earnings_yield_spread (M1, below), which feeds
#     the Graham framework, which feeds `score`, which is written to the daily
#     archive and to score_history. Making it live means a stock's Graham score
#     moves because the G-Sec moved 4bp. That is noise entering a stored score,
#     it breaks comparability with every archived snapshot, and it would require
#     a SCHEMA_VERSION bump and a reconcile run to do honestly.
#
# It also sits in the BM1 Gordon-growth denominator (r - g) further down, where
# r approaching g makes intrinsic value explode. macro_fetch.py's docstring has
# carried that boundary since 2026-07-28; it still stands.
#
# MONITORING, not maintenance: macro_read.rate_monitor(INDIA_10Y_BOND_RATE)
# reports the live operative 10Y against this constant and flags drift past
# macro_read.REEXAMINE_BP (100bp, pre-registered 2026-09-18 with the then-current
# 5.5bp drift already visible and stated). Exceeding it is not a failure and
# must not auto-update anything — it is the trigger to re-examine scoring
# sensitivity as its own decision, which is a sprint, not a config change.
# preflight Section H reports the drift on every run.
INDIA_10Y_BOND_RATE = 7.0
INDIA_10Y_BOND_RATE_AS_OF = "2026-07-01"
ADEQUATE_SIZE_INR = 2_000_000_000  # ₹200Cr (PPP-adjusted from Graham's $100M)
# COST_OF_CAPITAL_PROXY — SPECIFICATION OF THIS PROJECT, not a book number.
# Set 2026-07-29. Value UNCHANGED at 12.0; this edit only records why.
#
# Construction: Indian sovereign 10Y (~7.0) + ~5pp equity risk premium. That is
# a standard build, but the 5pp is a JUDGEMENT, not sourced from any of the
# seven books. None of Reilly & Brown, Graham, Dorsey or Buffett supplies an
# India cost-of-capital figure.
#
# DELIBERATELY NOT written as `INDIA_10Y_BOND_RATE + EQUITY_RISK_PREMIUM`.
# Expressing the derivation in code would be better documentation but would
# COUPLE this to the bond rate, and the roadmap bars wiring the observed 10Y
# until the Gordon singularity (496-503) is fixed. Coupling widens that fix's
# blast radius from one column to four, and at ~10 survivors per percentage
# point (below) a live sovereign drifting 50bps would silently move ~5
# holdings. Kept independent on purpose; the derivation lives here where it
# cannot propagate.
#
# Sensitivity, measured 2026-07-29 on 4,476 rows / 659 _tier1 survivors
# (cocp_sweep.py; dorsey_pass reproduces as score >= 6 at 100.0%):
#     proxy   survivors passing dorsey (>= 6 of 10)
#      10.0   309   (+30)
#      11.0   289   (+10)
#      12.0   279   <- adopted
#      13.0   270    (-9)
#      14.0   261   (-18)
# Smooth and roughly linear, ~10 survivors per pp. Load-bearing but NOT
# knife-edged: no defensible value in 10-14 yields a materially different
# portfolio. 12.0 also sits at p75 of dorsey_roic (12.89), admitting the top
# 28.2% on return-on-capital -- a coincidence, not a justification, but it
# indicates the value is not miscalibrated.
#
# NOT REPRICED. Nothing measured argues for a different number, and moving it
# would be choosing a value to change the portfolio, which is backwards.
# Revisit only with an external cost-of-capital source, not with a sweep.
#
# Consumed at FOUR sites, all comparing a percent-scale return:
#   551   buffett_value_creating_growth   best_return > PROXY
#   853   buffett_rational_allocation     roic > PROXY
#   1176  dorsey_buffett_score D5         droic > PROXY  (+1 of 10)
#   1195  dorsey_buffett_graded D5        same, graded
COST_OF_CAPITAL_PROXY = 12.0


# ─── Helpers ───

def _sf(val, default=None):
    """Safely convert to float. Returns default for NaN/None/inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _bs_row(sheet, names, col, default=None):
    """Try multiple row names on a balance sheet / financials DataFrame."""
    if sheet is None or sheet.empty:
        return default
    for name in names:
        if name in sheet.index:
            val = _sf(sheet.loc[name, col])
            if val is not None:
                return val
    return default


def _get_yearly_values(sheet, row_names, cols):
    """Extract a list of yearly values for a given row, most-recent-first."""
    vals = []
    for col in cols:
        vals.append(_bs_row(sheet, row_names, col))
    return vals


def _pct_growth(new, old):
    """Compute percentage growth. Returns None if not computable."""
    if new is None or old is None or old == 0:
        return None
    return (new / old - 1) * 100


# ─── 1. BALANCE SHEET HEALTH (14 columns) ───

# ── Frozen axis reference (Quality/Growth/Price/Safety), committed to repo ──
import os as _os, json as _json
try:
    from axis_scores import grade_row as _axis_grade
    with open(_os.path.join(_os.path.dirname(__file__), "axis_reference.json")) as _f:
        _AXIS_REF = _json.load(_f)
except Exception:
    _axis_grade, _AXIS_REF = None, None


def compute_axis_scores(data):
    """Relative axis profile vs the FROZEN Indian reference. Terminal scorer:
    deterministic from stored metrics + committed axis_reference.json.
    None-safe if the reference isn't present yet (columns emit None)."""
    s = _axis_grade(data, _AXIS_REF) if (_axis_grade and _AXIS_REF) else {}
    data["quality_axis"] = s.get("quality_axis")
    data["growth_axis"]  = s.get("growth_axis")
    data["price_axis"]   = s.get("price_axis")
    data["safety_axis"]  = s.get("safety_axis")


def compute_balance_sheet(data, info, bs, shares):
    """Compute balance sheet health metrics."""
    price = _sf(data.get("price"))
    revenue_y0 = _sf(data.get("revenue_y0"))

    # D1: Adequate Size
    data["graham_adequate_size"] = bool(revenue_y0 and revenue_y0 >= ADEQUATE_SIZE_INR)

    # D2: Current Ratio Pass (≥ 2.0 for defensive)
    cr = _sf(data.get("current_ratio"))
    data["graham_current_ratio_pass"] = bool(cr and cr >= 2.0)

    # graham_bvps reads info["bookValue"] and depends on NEITHER bs NOR shares.
    # It was nulled by a gate that has nothing to do with it.
    _bv = _sf(info.get("bookValue"))
    data["graham_bvps"] = round(_bv, 2) if _bv else None

    if bs is None or bs.empty:
        # No balance sheet -> nothing below is computable. graham_bvps survives.
        for col in ["graham_ltd_vs_nca", "graham_net_current_assets", "graham_ncav_per_share",
                     "graham_ncav_ratio", "graham_price_to_ntav",
                     "graham_net_cash", "lynch_net_cash_per_share",
                     "dorsey_financial_leverage", "dorsey_interest_coverage",
                     "dorsey_quick_ratio", "dorsey_clean_balance_sheet"]:
            data[col] = None
        return

    # A missing share count is NOT a missing balance sheet. Only the four
    # PER-SHARE columns become uncomputable; the other seven read bs/info
    # directly. Nulling all twelve destroyed the whole balance-sheet picture
    # for companies whose balance sheet was fully readable -- 180 rows fire
    # this, 97.8% score 0 against a 53.0% baseline. Proof case 3PLAND.NS.
    _has_sh = shares is not None and shares > 0

    cols = sorted(bs.columns)
    latest = cols[-1]

    ca = _bs_row(bs, ["Current Assets", "Total Current Assets"], latest)
    cl = _bs_row(bs, ["Current Liabilities", "Total Current Liabilities"], latest)
    total_assets = _bs_row(bs, ["Total Assets"], latest)
    total_debt = _sf(info.get("totalDebt"))
    total_cash = _sf(info.get("totalCash"))
    equity = _bs_row(bs, ["Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest"], latest)
    ltd = _bs_row(bs, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], latest)
    inventory = _bs_row(bs, ["Inventory"], latest)
    intangibles = _bs_row(bs, ["Intangible Assets", "Other Intangible Assets"], latest, 0)
    goodwill = _bs_row(bs, ["Goodwill"], latest, 0)
    bvps = _sf(info.get("bookValue"))

    nca = (ca - cl) if (ca is not None and cl is not None) else None

    # D3: Long-Term Debt vs NCA
    if ltd is not None and nca is not None:
        data["graham_ltd_vs_nca"] = bool(ltd <= nca)
    else:
        data["graham_ltd_vs_nca"] = None

    # F1: Net Current Assets
    data["graham_net_current_assets"] = round(nca, 2) if nca is not None else None

    # current_ratio fallback (info-first): from the balance sheet when info lacked it
    if data.get("current_ratio") is None and ca and cl and cl != 0:
        data["current_ratio"] = round(ca / cl, 2)

    # N1: NCAV per share
    total_liab = _bs_row(bs, ["Total Liabilities Net Minority Interest", "Total Liab"], latest)
    if ca is not None and total_liab is not None and _has_sh:
        ncav = ca - total_liab
        data["graham_ncav_per_share"] = round(ncav / shares, 2)
    else:
        data["graham_ncav_per_share"] = None

    # N2: NCAV Ratio
    ncav_ps = _sf(data.get("graham_ncav_per_share"))
    if ncav_ps and ncav_ps > 0 and price:
        data["graham_ncav_ratio"] = round(price / ncav_ps, 2)
    else:
        data["graham_ncav_ratio"] = None

    # F3: Book Value per Share — already set above the bs gate, before any
    # early return could destroy it. Left here as a no-op for traceability.

    # E5: Price to Net Tangible Assets
    if total_assets is not None and total_liab is not None:
        ntav = total_assets - (intangibles or 0) - (goodwill or 0) - total_liab
        ntav_ps = ntav / shares if _has_sh else None
        if ntav_ps and ntav_ps > 0 and price:
            data["graham_price_to_ntav"] = round(price / ntav_ps, 2)
        else:
            data["graham_price_to_ntav"] = None
    else:
        data["graham_price_to_ntav"] = None

    # I6: Net Cash
    if total_cash is not None and total_debt is not None:
        net_cash = total_cash - total_debt
        data["graham_net_cash"] = round(net_cash, 2)
        # graham_net_cash needs no share count; only the per-share form does.
        data["lynch_net_cash_per_share"] = (round(net_cash / shares, 2)
                                            if _has_sh else None)
    else:
        data["graham_net_cash"] = None
        data["lynch_net_cash_per_share"] = None

    # F1 Dorsey: Financial Leverage
    if total_assets and equity and equity > 0:
        data["dorsey_financial_leverage"] = round(total_assets / equity, 2)
    else:
        data["dorsey_financial_leverage"] = None

    # F4: Quick Ratio
    if ca is not None and cl is not None and cl > 0:
        inv = inventory or 0
        data["dorsey_quick_ratio"] = round((ca - inv) / cl, 2)
    else:
        data["dorsey_quick_ratio"] = None

    # T5: Clean Balance Sheet
    de = _sf(data.get("de"))
    data["dorsey_clean_balance_sheet"] = bool(de is not None and de <= 100)  # de is in %, ≤100% = D/E ≤ 1.0

    # F2: Interest Coverage — computed in earnings quality (needs EBIT from income stmt)
    data["dorsey_interest_coverage"] = None  # placeholder, filled later


# ─── 2. EARNINGS QUALITY (12 columns) ───

def compute_earnings_quality(data, info, income_stmt, cashflow, bs, shares):
    """Compute earnings quality metrics."""
    # Gather NI values
    ni = [_sf(data.get(f"net_income_y{i}")) for i in range(4)]
    valid_ni = [v for v in ni if v is not None]

    # D4: Earnings Stability
    if len(valid_ni) >= 4:
        data["graham_earnings_stable_4y"] = all(v > 0 for v in valid_ni)
    else:
        data["graham_earnings_stable_4y"] = False  # Fail conservatively

    # Q1: Average EPS
    if valid_ni and shares and shares > 0:
        avg_ni = sum(valid_ni) / len(valid_ni)
        data["graham_avg_eps_4y"] = round(avg_ni / shares, 2)
    else:
        data["graham_avg_eps_4y"] = None

    # Q2: EPS Coefficient of Variation
    if len(valid_ni) >= 3 and shares and shares > 0:
        eps_vals = [v / shares for v in valid_ni]
        mean_eps = sum(eps_vals) / len(eps_vals)
        if mean_eps != 0:
            std_eps = (sum((e - mean_eps) ** 2 for e in eps_vals) / len(eps_vals)) ** 0.5
            data["graham_eps_cv"] = round(abs(std_eps / mean_eps), 4)
        else:
            data["graham_eps_cv"] = None
    else:
        data["graham_eps_cv"] = None

    # D6: EPS Growth (3Y-avg method)
    if ni[0] is not None and ni[1] is not None and ni[2] is not None and ni[3] is not None:
        avg_recent = (ni[0] + ni[1]) / 2
        avg_older = (ni[2] + ni[3]) / 2
        if avg_older > 0:
            data["graham_eps_growth_pct_4y"] = round((avg_recent / avg_older - 1) * 100, 2)
        else:
            data["graham_eps_growth_pct_4y"] = None
    else:
        data["graham_eps_growth_pct_4y"] = None

    # ── Cash flow based quality metrics ──
    if cashflow is not None and not cashflow.empty:
        cf_cols = sorted(cashflow.columns)
        latest_cf = cf_cols[-1]

        ocf_vals = _get_yearly_values(cashflow,
            ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"],
            cf_cols[-4:] if len(cf_cols) >= 4 else cf_cols)
        capex_vals = _get_yearly_values(cashflow,
            ["Capital Expenditure", "Purchase Of PPE"],
            cf_cols[-4:] if len(cf_cols) >= 4 else cf_cols)

        ocf_y0 = ocf_vals[-1] if ocf_vals else None
        capex_y0 = capex_vals[-1] if capex_vals else None
        fcf_y0 = (ocf_y0 + capex_y0) if (ocf_y0 is not None and capex_y0 is not None) else None  # CapEx is negative in yfinance

        total_assets = _bs_row(bs, ["Total Assets"], sorted(bs.columns)[-1]) if (bs is not None and not bs.empty) else None

        # S6: Accruals Ratio (REPLACES cash_conversion)
        if ni[0] is not None and ocf_y0 is not None and total_assets and total_assets > 0:
            data["schilit_accruals_ratio"] = round((ni[0] - ocf_y0) / total_assets, 4)
        else:
            data["schilit_accruals_ratio"] = None

        # S7: CFO / NI ratio
        if ni[0] is not None and ni[0] > 0 and ocf_y0 is not None:
            data["schilit_cfo_ni_ratio"] = round(ocf_y0 / ni[0], 2)
        else:
            data["schilit_cfo_ni_ratio"] = None

        # S8: FCF / NI ratio
        if ni[0] is not None and ni[0] > 0 and fcf_y0 is not None:
            data["schilit_fcf_ni_ratio"] = round(fcf_y0 / ni[0], 2)
        else:
            data["schilit_fcf_ni_ratio"] = None

        # M1: Excess Cash Margin
        oi_y0 = None
        if income_stmt is not None and not income_stmt.empty:
            is_cols = sorted(income_stmt.columns)
            oi_y0 = _bs_row(income_stmt, ["Operating Income", "EBIT"], is_cols[-1])
        rev_y0 = _sf(data.get("revenue_y0"))
        if ocf_y0 is not None and oi_y0 is not None and rev_y0 and rev_y0 > 0:
            data["mulford_ecm"] = round((ocf_y0 - oi_y0) / rev_y0 * 100, 2)
        else:
            data["mulford_ecm"] = None

        # M2: ECM Trend
        if len(cf_cols) >= 2 and income_stmt is not None and not income_stmt.empty:
            is_cols = sorted(income_stmt.columns)
            prev_ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-2])
            prev_oi = _bs_row(income_stmt, ["Operating Income", "EBIT"], is_cols[-2]) if len(is_cols) >= 2 else None
            rev_y1 = _sf(data.get("revenue_y1"))
            if prev_ocf is not None and prev_oi is not None and rev_y1 and rev_y1 > 0:
                ecm_prev = (prev_ocf - prev_oi) / rev_y1 * 100
                ecm_curr = _sf(data.get("mulford_ecm"))
                if ecm_curr is not None:
                    data["mulford_ecm_trend"] = round(ecm_curr - ecm_prev, 2)
                else:
                    data["mulford_ecm_trend"] = None
            else:
                data["mulford_ecm_trend"] = None
        else:
            data["mulford_ecm_trend"] = None

        # M3: Cash Margin
        if ocf_y0 is not None and rev_y0 and rev_y0 > 0:
            data["mulford_cash_margin"] = round(ocf_y0 / rev_y0 * 100, 2)
        else:
            data["mulford_cash_margin"] = None

        # M4: OCF / Operating Income
        if ocf_y0 is not None and oi_y0 is not None and oi_y0 > 0:
            data["mulford_ocf_oi_ratio"] = round(ocf_y0 / oi_y0, 2)
        else:
            data["mulford_ocf_oi_ratio"] = None

        # T2: Consistent CFO
        valid_ocf = [v for v in ocf_vals if v is not None]
        data["dorsey_consistent_cfo"] = bool(len(valid_ocf) >= 3 and all(v > 0 for v in valid_ocf))

        # Interest Coverage (needs EBIT from income_stmt)
        if income_stmt is not None and not income_stmt.empty:
            is_cols = sorted(income_stmt.columns)
            ebit = _bs_row(income_stmt, ["EBIT", "Operating Income"], is_cols[-1])
            int_exp = _bs_row(income_stmt, ["Interest Expense", "Interest Expense Non Operating", "Net Interest Income"], is_cols[-1])
            if ebit is not None and int_exp is not None and int_exp != 0:
                data["dorsey_interest_coverage"] = round(abs(ebit / int_exp), 2)

    else:
        for col in ["schilit_accruals_ratio", "schilit_cfo_ni_ratio", "schilit_fcf_ni_ratio",
                     "mulford_ecm", "mulford_ecm_trend", "mulford_cash_margin",
                     "mulford_ocf_oi_ratio", "dorsey_consistent_cfo"]:
            data[col] = None


# ─── 3. VALUATION (14 columns) ───

def compute_valuation(data, info, income_stmt, cashflow, bs, shares):
    """Compute valuation metrics."""
    price = _sf(data.get("price"))
    pe = _sf(data.get("pe"))
    pb = _sf(data.get("pb"))
    eps = _sf(data.get("eps"))
    avg_eps = _sf(data.get("graham_avg_eps_4y"))
    bvps = _sf(data.get("graham_bvps"))
    ni_cagr = _sf(data.get("ni_cagr_3y"))
    div_yield = _sf(data.get("dividend_yield"))
    market_cap = _sf(info.get("marketCap"))
    total_debt = _sf(info.get("totalDebt"))
    total_cash = _sf(info.get("totalCash"))

    # D7a: PE on average EPS. NB the column is named graham_pe_3y_avg but the
    # input is graham_avg_eps_4y — a FOUR-year average. The name is legacy and
    # kept for archive compatibility; the value has always been 4y.
    if price and avg_eps and avg_eps > 0:
        data["graham_pe_3y_avg"] = round(price / avg_eps, 2)
    else:
        data["graham_pe_3y_avg"] = None
    # Local alias so D7c below reads on the averaged basis, not trailing `pe`.
    pe_avg_eps = _sf(data.get("graham_pe_3y_avg"))

    # ── D7c / D8 on AVERAGED earnings (W2 step 7) ────────────────────────
    # Graham's D7 is stated on AVERAGE earnings, and the averaging is not a
    # detail — it IS his anti-cyclical-value-trap device. Trailing PE at the top
    # of a cycle reads cheap precisely when earnings are about to fall. Both
    # tests previously used trailing eps/pe; both now use graham_avg_eps_4y via
    # graham_pe_3y_avg, computed just above.
    #
    # NO FALLBACK to trailing when the average is missing. Substituting the
    # unaveraged number would reintroduce the exact error the change removes,
    # and it is not needed: coverage IMPROVES on net (see below).
    #
    # ── MEASURED EFFECT, 4,477 fresh rows, pre-change snapshot ───────────
    # D8 PASSES its falsification test: pass->fail rate 4.8% on declared
    # cyclicals vs 2.1% elsewhere — 2.3x, i.e. it strips more false cheapness
    # from cyclicals, which is what the peak-earnings argument predicts.
    #
    # D7c DOES NOT: 4.8% cyclical vs 5.9% non-cyclical — it bites slightly LESS
    # on cyclicals. Shipped on BOOK FIDELITY ALONE, not on demonstrated efficacy.
    # Two readings we cannot separate: (a) this snapshot contains no cyclical
    # peak (Indian steel revenue was ~flat 2022-26), so a cross-sectional test
    # is underpowered for a peak-detector; (b) D7c is a PRODUCT with P/B, so
    # fixing one factor dilutes the signal. Recorded rather than glossed.
    #
    # The larger win is D8 COVERAGE, and it is a fail-closed bug, not a
    # refinement: trailing eps>0 held for 44.0% of rows, averaged eps>0 for
    # 74.4%. A firm with one loss year had no trailing EPS, so graham_number was
    # None and D8 scored 0 — reading as "fails Graham's price test" when it
    # meant "not computable". 1,368 rows GAIN the test; 98 lose it.
    # Net repricing: 694 rows (15.5%) move on graham_defensive_score, of which
    # only 23 currently score >= 4.
    if pe_avg_eps and pe_avg_eps > 0 and pb and pb > 0:
        data["graham_pe_pb_composite"] = round(pe_avg_eps * pb, 2)
    else:
        data["graham_pe_pb_composite"] = None

    # D8: Graham Number on average EPS. Conventionally quoted with trailing EPS,
    # but the 22.5 constant is 15 P/E x 1.5 P/B and inherits D7's averaged-
    # earnings basis, so trailing was the inconsistency.
    if avg_eps and avg_eps > 0 and bvps and bvps > 0:
        data["graham_number"] = round((22.5 * avg_eps * bvps) ** 0.5, 2)
    else:
        data["graham_number"] = None

    # M1: Earnings Yield Spread vs bond rate
    if pe and pe > 0:
        ey = 1.0 / pe * 100
        data["graham_earnings_yield_spread"] = round(ey - INDIA_10Y_BOND_RATE, 2)
    else:
        data["graham_earnings_yield_spread"] = None

    # M2: Graham Intrinsic Value
    g = min(ni_cagr, 25) if (ni_cagr is not None and ni_cagr > 0) else 0
    if avg_eps and avg_eps > 0:
        data["graham_intrinsic_value"] = round(avg_eps * (8.5 + 2 * g), 2)
    else:
        data["graham_intrinsic_value"] = None

    # M3: Graham Margin of Safety
    giv = _sf(data.get("graham_intrinsic_value"))
    if giv and giv > 0 and price and price > 0:
        data["graham_margin_of_safety_pct"] = round((giv - price) / price * 100, 2)
    else:
        data["graham_margin_of_safety_pct"] = None

    # G2: Greenblatt Earnings Yield (EBIT / EV)
    ebit = None
    if income_stmt is not None and not income_stmt.empty:
        is_cols = sorted(income_stmt.columns)
        ebit = _bs_row(income_stmt, ["EBIT", "Operating Income"], is_cols[-1])
    ev = None
    if market_cap is not None:
        td = total_debt or 0
        tc = total_cash or 0
        ev = market_cap + td - tc
    if ebit and ebit > 0 and ev and ev > 0:
        data["greenblatt_earnings_yield"] = round(ebit / ev * 100, 2)
    else:
        data["greenblatt_earnings_yield"] = None

    # G3: Combined rank — computed at universe level in score_all_stocks()
    data["greenblatt_combined_rank"] = None  # Placeholder

    # P1: Lynch PEG
    if pe and pe > 0 and ni_cagr and ni_cagr > 0:
        data["lynch_peg"] = round(pe / ni_cagr, 2)
    else:
        data["lynch_peg"] = None

    # P2: Lynch PEG Adjusted (dividend-adjusted)
    dy_pct = (div_yield * 100) if div_yield else 0
    if pe and pe > 0 and ni_cagr and ni_cagr > 0:
        data["lynch_peg_adjusted"] = round((ni_cagr + dy_pct) / pe, 2)
    else:
        data["lynch_peg_adjusted"] = None

    # C3: Lynch Cash-Adjusted PE
    net_cash_ps = _sf(data.get("lynch_net_cash_per_share"))
    if price and eps and eps > 0 and net_cash_ps is not None:
        adj_price = price - max(net_cash_ps, 0)  # Only subtract if net cash positive
        if adj_price > 0:
            data["lynch_cash_adjusted_pe"] = round(adj_price / eps, 2)
        else:
            data["lynch_cash_adjusted_pe"] = None
    else:
        data["lynch_cash_adjusted_pe"] = None

    # V1 Dorsey: Cash Return (FCF / EV)
    if cashflow is not None and not cashflow.empty and ev and ev > 0:
        cf_cols = sorted(cashflow.columns)
        ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-1])
        capex = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], cf_cols[-1])
        if ocf is not None and capex is not None:
            fcf = ocf + capex  # capex is negative
            data["dorsey_cash_return"] = round(fcf / ev * 100, 2)
        else:
            data["dorsey_cash_return"] = None
    else:
        data["dorsey_cash_return"] = None

    # BM1: Buffett Intrinsic Value (DCF perpetuity)
    ocf_val = None
    capex_val = None
    if cashflow is not None and not cashflow.empty:
        cf_cols = sorted(cashflow.columns)
        ocf_val = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-1])
        capex_val = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], cf_cols[-1])

    if ocf_val is not None and capex_val is not None and shares and shares > 0:
        oe_ps = (ocf_val + capex_val) / shares  # Owner earnings per share
        g_rate = min(ni_cagr / 100, 0.15) if (ni_cagr and ni_cagr > 0) else 0.03
        r_rate = INDIA_10Y_BOND_RATE / 100
        if r_rate > g_rate:
            data["buffett_intrinsic_value"] = round(oe_ps * (1 + g_rate) / (r_rate - g_rate), 2)
        else:
            data["buffett_intrinsic_value"] = None
    else:
        data["buffett_intrinsic_value"] = None

    # BM2: Buffett Margin of Safety
    biv = _sf(data.get("buffett_intrinsic_value"))
    if biv and biv > 0 and price and price > 0:
        data["buffett_margin_of_safety_pct"] = round((biv - price) / biv * 100, 2)
    else:
        data["buffett_margin_of_safety_pct"] = None


# ─── 4. GROWTH TRAJECTORY (5 columns) ───

def compute_growth(data, income_stmt, shares):
    """Compute growth trajectory metrics."""
    ni_cagr = _sf(data.get("ni_cagr_3y"))

    # P3: Lynch Growth Flag
    if ni_cagr is None:
        data["lynch_growth_flag"] = "unknown"
    elif ni_cagr > 50:
        data["lynch_growth_flag"] = "dangerous"
    elif ni_cagr > 25:
        data["lynch_growth_flag"] = "suspicious"
    elif ni_cagr >= 20:
        data["lynch_growth_flag"] = "ideal"
    elif ni_cagr >= 12:
        data["lynch_growth_flag"] = "acceptable"
    else:
        data["lynch_growth_flag"] = "slow"

    # I7: Growth Acceleration
    ni = [_sf(data.get(f"net_income_y{i}")) for i in range(4)]
    if ni[0] is not None and ni[1] is not None and ni[1] > 0 and ni[2] is not None and ni[2] > 0:
        recent_growth = (ni[0] / ni[1] - 1) * 100
        prior_growth = (ni[1] / ni[2] - 1) * 100
        data["lynch_growth_acceleration"] = round(recent_growth - prior_growth, 2)
    else:
        data["lynch_growth_acceleration"] = None

    # I6 Buffett: Value-creating growth
    ni_cagr_val = ni_cagr if ni_cagr else 0
    # We need ROIC — use greenblatt_roic if already computed, else use ROE as proxy
    roic = _sf(data.get("greenblatt_roic")) or _sf(data.get("dorsey_roic"))
    roe = _sf(data.get("roe"))
    best_return = roic or (roe * 100 if roe else None)  # roe is decimal in data
    if best_return and ni_cagr_val > 0:
        data["buffett_value_creating_growth"] = bool(best_return > COST_OF_CAPITAL_PROXY)
    else:
        data["buffett_value_creating_growth"] = None

    # GI3: Average EBIT (4Y)
    if income_stmt is not None and not income_stmt.empty:
        is_cols = sorted(income_stmt.columns)
        ebit_vals = []
        for col in is_cols[-4:]:
            ebit = _bs_row(income_stmt, ["EBIT", "Operating Income"], col)
            if ebit is not None:
                ebit_vals.append(ebit)
        if ebit_vals:
            data["greenblatt_ebit_avg_4y"] = round(sum(ebit_vals) / len(ebit_vals), 2)
        else:
            data["greenblatt_ebit_avg_4y"] = None
    else:
        data["greenblatt_ebit_avg_4y"] = None

    # E4: Enterprising earnings growing
    if ni[0] is not None and ni[1] is not None:
        data["graham_ent_earnings_growing"] = bool(ni[0] > ni[1])
    else:
        data["graham_ent_earnings_growing"] = None


# ─── 5. MOAT DURABILITY (8 columns) ───

def compute_moat(data, info, income_stmt, bs, cashflow, shares):
    # Default here, NOT in compute_manipulation_flags. That function runs AFTER
    # compute_moat and an unconditional default there resets the flag on every
    # row -- which is exactly what happened on the first v5 run: 226 expected
    # True, 0 observed. Defaults must be set before the branch that overrides
    # them, never after.
    data["greenblatt_capital_nonpositive"] = False
    """Compute moat durability metrics."""
    if income_stmt is None or income_stmt.empty:
        for col in ["greenblatt_roic", "greenblatt_roic_trend", "dorsey_roic",
                     "dorsey_fcf_margin", "dorsey_roe_consistent", "dorsey_roa",
                     "dorsey_pb_roe_signal", "buffett_roe_unleveraged"]:
            data[col] = None
        return

    is_cols = sorted(income_stmt.columns)
    latest_is = is_cols[-1]
    ebit = _bs_row(income_stmt, ["EBIT", "Operating Income"], latest_is)
    revenue = _bs_row(income_stmt, ["Total Revenue"], latest_is)

    # G1: Greenblatt ROIC = EBIT / (NWC + Net Fixed Assets)
    if bs is not None and not bs.empty and ebit is not None:
        bs_cols = sorted(bs.columns)
        latest_bs = bs_cols[-1]
        ca = _bs_row(bs, ["Current Assets", "Total Current Assets"], latest_bs)
        cl = _bs_row(bs, ["Current Liabilities", "Total Current Liabilities"], latest_bs)
        cash = 0  # V2 — see the v6 -> v7 schema note. NOT Greenblatt's denominator.
        st_debt = _bs_row(bs, ["Current Debt", "Short Long Term Debt", "Current Debt And Capital Lease Obligation"], latest_bs, 0)
        ppe = _bs_row(bs, ["Net PPE", "Property Plant Equipment Net", "Net Property Plant And Equipment"], latest_bs, 0)
        gw = _bs_row(bs, ["Goodwill"], latest_bs, 0)
        intang = _bs_row(bs, ["Intangible Assets", "Other Intangible Assets"], latest_bs, 0)

        if ca is not None and cl is not None:
            nwc = (ca - cash) - (cl - st_debt)
            tangible_capital = nwc + ppe
            if tangible_capital and tangible_capital > 0:
                data["greenblatt_roic"] = round(ebit / tangible_capital * 100, 2)
            else:
                # INAPPLICABILITY, not missing data. EBIT over a non-positive
                # denominator is not a return on capital in either direction:
                # a profitable business gets a NEGATIVE roic, and a loss-making
                # one gets a POSITIVE one. Greenblatt excludes financials on
                # exactly this reasoning -- the ratio is meaningless for that
                # balance-sheet shape -- and says nothing about negative
                # tangible capital anywhere in the book (checked: the appendix
                # ROC definition, the screening steps, and every occurrence of
                # "negative"). His demonstrated policy where the formula cannot
                # price something is to EXCLUDE, never to impute. So this
                # abstains rather than failing.
                #
                # The flag is required because `greenblatt_roic is None` has
                # THREE causes -- this one, a missing ca/cl, and no EBIT -- and
                # only this one is inapplicability. One flag carrying two
                # meanings is the W2 root-cause defect; this keeps them apart.
                #
                # Distress protection is NOT this metric's job. Measured
                # 2026-07-28: of 226 rows with tc <= 0, only 18 survive _tier1
                # and 17 of those have positive EBIT. _tier1 and the
                # Schilit/Mulford gate already strip the rest.
                data["greenblatt_roic"] = None
                data["greenblatt_capital_nonpositive"] = True
        else:
            data["greenblatt_roic"] = None
    else:
        data["greenblatt_roic"] = None

    # GI4: ROIC Trend (latest vs average)
    roic_vals = []
    if bs is not None and not bs.empty:
        bs_cols = sorted(bs.columns)
        for i, bc in enumerate(bs_cols[-4:]):
            ic = is_cols[-(4-i)] if len(is_cols) >= (4-i) else None
            if ic is None:
                continue
            e = _bs_row(income_stmt, ["EBIT", "Operating Income"], ic)
            ca_i = _bs_row(bs, ["Current Assets", "Total Current Assets"], bc)
            cl_i = _bs_row(bs, ["Current Liabilities", "Total Current Liabilities"], bc)
            ppe_i = _bs_row(bs, ["Net PPE", "Property Plant Equipment Net", "Net Property Plant And Equipment"], bc, 0)
            if e is not None and ca_i is not None and cl_i is not None:
                tc = (ca_i - cl_i) + ppe_i
                if tc and tc > 0:
                    roic_vals.append(e / tc * 100)
    if len(roic_vals) >= 2:
        avg_roic = sum(roic_vals) / len(roic_vals)
        if avg_roic != 0:
            data["greenblatt_roic_trend"] = round(roic_vals[-1] / avg_roic, 2)
        else:
            data["greenblatt_roic_trend"] = None
    else:
        data["greenblatt_roic_trend"] = None

    # D5 Dorsey: ROIC (after-tax)
    if ebit is not None and bs is not None and not bs.empty:
        bs_cols = sorted(bs.columns)
        latest_bs = bs_cols[-1]
        ta = _bs_row(bs, ["Total Assets"], latest_bs)
        # Approximate tax rate from NI / pretax income, or default 25%
        ni_y0 = _sf(data.get("net_income_y0"))
        pretax = _bs_row(income_stmt, ["Pretax Income", "Income Before Tax"], latest_is)
        tax_rate = 0.25  # default
        if ni_y0 and pretax and pretax > 0:
            tax_rate = max(0, 1 - ni_y0 / pretax)
        nopat = ebit * (1 - tax_rate)
        # Invested Capital = Total Assets - non-interest-bearing CL - excess cash
        cl = _bs_row(bs, ["Current Liabilities", "Total Current Liabilities"], latest_bs)
        cash = _sf(info.get("totalCash")) or 0
        if ta and cl:
            invested_cap = ta - cl  # Simplified
            if invested_cap > 0:
                data["dorsey_roic"] = round(nopat / invested_cap * 100, 2)
            else:
                data["dorsey_roic"] = None
        else:
            data["dorsey_roic"] = None
    else:
        data["dorsey_roic"] = None

    # D1: FCF Margin
    if cashflow is not None and not cashflow.empty and revenue and revenue > 0:
        cf_cols = sorted(cashflow.columns)
        ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-1])
        capex = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], cf_cols[-1])
        if ocf is not None and capex is not None:
            fcf = ocf + capex
            data["dorsey_fcf_margin"] = round(fcf / revenue * 100, 2)
        else:
            data["dorsey_fcf_margin"] = None
    else:
        data["dorsey_fcf_margin"] = None

    # D3: ROE Consistent (≥15% in ≥3 of 4 years)
    roe_vals = [_sf(data.get(f"roe_y{i}")) for i in range(4)]
    valid_roe = [v for v in roe_vals if v is not None]
    if len(valid_roe) >= 3:
        data["dorsey_roe_consistent"] = sum(1 for v in valid_roe if v >= 15) >= 3
    else:
        data["dorsey_roe_consistent"] = None

    # Statement fallbacks (info-first): fill only when yfinance info lacked the field.
    # roe_y0 is % -> info roe is a fraction; net_income_y0 / total-assets & revenue for the rest.
    if data.get("roe") is None:
        _r0 = _sf(data.get("roe_y0"))
        if _r0 is not None:
            data["roe"] = round(_r0 / 100.0, 4)
    if data.get("profit_margin") is None:
        _ni, _rev = _sf(data.get("net_income_y0")), _sf(data.get("revenue_y0"))
        if _ni is not None and _rev and _rev != 0:
            data["profit_margin"] = round(_ni / _rev, 4)

    # D4: ROA
    data["dorsey_roa"] = _sf(info.get("returnOnAssets"))
    if data["dorsey_roa"] is not None:
        data["dorsey_roa"] = round(data["dorsey_roa"] * 100, 2)
    else:
        _ni = _sf(data.get("net_income_y0"))
        _ta = _bs_row(bs, ["Total Assets"], sorted(bs.columns)[-1]) if (bs is not None and not bs.empty) else None
        if _ni is not None and _ta and _ta != 0:
            data["dorsey_roa"] = round(_ni / _ta * 100, 2)   # % , matching the info path

    # V3: P/B + ROE signal
    pb = _sf(data.get("pb"))
    roe = _sf(data.get("roe"))
    if pb and roe:
        data["dorsey_pb_roe_signal"] = bool(pb < 1.5 and roe > 0.15)
    else:
        data["dorsey_pb_roe_signal"] = None

    # BF1: Buffett ROE Unleveraged
    de = _sf(data.get("de"))
    if roe and de is not None:
        data["buffett_roe_unleveraged"] = bool(roe >= 0.15 and de <= 50)  # de in %, 50% = D/E 0.5
    else:
        data["buffett_roe_unleveraged"] = None


# ─── 6. DIVIDEND QUALITY (4 columns) ───

def compute_dividends(data, stock, income_stmt, shares):
    """Compute dividend quality metrics."""
    # Consecutive dividend years
    try:
        divs = stock.dividends
        if divs is not None and not divs.empty:
            years_with_divs = set()
            for dt in divs.index:
                if divs[dt] > 0:
                    years_with_divs.add(dt.year)
            if years_with_divs:
                current_year = datetime.now().year
                consec = 0
                for yr in range(current_year, current_year - 50, -1):
                    if yr in years_with_divs:
                        consec += 1
                    else:
                        break
                data["dividend_consecutive_years"] = consec
            else:
                data["dividend_consecutive_years"] = 0
        else:
            data["dividend_consecutive_years"] = 0
    except Exception:
        data["dividend_consecutive_years"] = 0

    # Payout Ratio
    ni_y0 = _sf(data.get("net_income_y0"))
    div_yield = _sf(data.get("dividend_yield"))
    market_cap = _sf(data.get("market_cap"))
    if ni_y0 and ni_y0 > 0 and div_yield and market_cap:
        total_divs = div_yield * market_cap
        data["graham_payout_ratio"] = round(total_divs / ni_y0 * 100, 2)
    else:
        data["graham_payout_ratio"] = None

    # E3: Has Dividend
    data["graham_ent_has_dividend"] = bool(div_yield and div_yield > 0)

    # I9: Deep Value Flag
    pb = _sf(data.get("pb"))
    data["graham_deep_value_flag"] = bool(pb and pb <= 0.67 and ni_y0 and ni_y0 > 0)


# ─── 7. MANAGEMENT QUALITY (5 columns) ───

def compute_management(data, info, income_stmt, cashflow, bs, shares, stock):
    """Compute management quality metrics (Buffett tenets)."""
    price = _sf(data.get("price"))
    market_cap = _sf(info.get("marketCap"))

    # BF2: Owner Earnings per share
    if cashflow is not None and not cashflow.empty and shares and shares > 0:
        cf_cols = sorted(cashflow.columns)
        ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-1])
        capex = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], cf_cols[-1])
        if ocf is not None and capex is not None:
            data["buffett_owner_earnings_ps"] = round((ocf + capex) / shares, 2)
        else:
            data["buffett_owner_earnings_ps"] = None
    else:
        data["buffett_owner_earnings_ps"] = None

    # BF4: One-Dollar Premise
    # ΔMarketCap / ΣRetainedEarnings over available years
    ni_vals = [_sf(data.get(f"net_income_y{i}")) for i in range(4)]
    valid_ni = [v for v in ni_vals if v is not None]
    div_yield = _sf(data.get("dividend_yield"))
    payout = _sf(data.get("graham_payout_ratio"))

    if len(valid_ni) >= 3 and market_cap and price:
        # Estimate retained earnings: NI × (1 - payout_ratio)
        retention = 1.0 - (payout / 100 if payout else 0.3)  # Default 30% payout
        retained = sum(v * retention for v in valid_ni)
        # Estimate market cap change: approximate from price change
        # Use earliest available year's NI/equity to approximate old price
        try:
            hist = stock.history(period="4y")
            if hist is not None and not hist.empty and len(hist) > 250:
                old_price = float(hist["Close"].iloc[0])
                delta_mcap = (price - old_price) * shares
                if retained > 0:
                    data["buffett_one_dollar_test"] = round(delta_mcap / retained, 2)
                else:
                    data["buffett_one_dollar_test"] = None
            else:
                data["buffett_one_dollar_test"] = None
        except Exception:
            data["buffett_one_dollar_test"] = None
    else:
        data["buffett_one_dollar_test"] = None

    # BG1: Rational Allocation
    roic = _sf(data.get("greenblatt_roic")) or _sf(data.get("dorsey_roic"))
    dilution = _sf(data.get("dorsey_share_dilution_pct"))
    payout_r = _sf(data.get("graham_payout_ratio"))
    if roic is not None:
        if roic > COST_OF_CAPITAL_PROXY:
            # High ROIC — reinvesting is rational
            data["buffett_rational_allocation"] = True
        elif payout_r and payout_r > 40:
            # Low ROIC but returning cash — rational
            data["buffett_rational_allocation"] = True
        elif dilution is not None and dilution < 0:
            # Low ROIC but buying back shares — rational
            data["buffett_rational_allocation"] = True
        else:
            data["buffett_rational_allocation"] = False
    else:
        data["buffett_rational_allocation"] = None

    # T8 Dorsey: Share Dilution — annualized growth in share count (buyback = negative)
    if bs is not None and not bs.empty:
        bs_cols = sorted(bs.columns)                      # oldest -> newest
        share_names = ["Ordinary Shares Number", "Share Issued", "Common Stock"]
        sh_new = _bs_row(bs, share_names, bs_cols[-1])
        sh_old = _bs_row(bs, share_names, bs_cols[0])
        yrs = len(bs_cols) - 1
        if sh_new and sh_old and sh_new > 0 and sh_old > 0 and yrs >= 1:
            data["dorsey_share_dilution_pct"] = round(((sh_new / sh_old) ** (1.0 / yrs) - 1.0) * 100, 3)
        else:
            data["dorsey_share_dilution_pct"] = None     # no share-count history in bs
    else:
        data["dorsey_share_dilution_pct"] = None

    # T1: Has Operating Profit
    if income_stmt is not None and not income_stmt.empty:
        is_cols = sorted(income_stmt.columns)
        has_op = False
        for col in is_cols:
            oi = _bs_row(income_stmt, ["Operating Income", "EBIT"], col)
            if oi is not None and oi > 0:
                has_op = True
                break
        data["dorsey_has_operating_profit"] = has_op
    else:
        data["dorsey_has_operating_profit"] = None


# ─── 8. MANIPULATION FLAGS (15 columns) ───

def compute_manipulation_flags(data, income_stmt, bs, cashflow):
    """Compute Schilit + Dorsey + Lynch manipulation red flags."""

    # Initialize all to None
    flag_cols = ["schilit_dso", "schilit_ar_revenue_divergence", "schilit_capex_depr_ratio",
                 "schilit_dsi", "schilit_inventory_revenue_div", "schilit_wc_cffo_pct",
                 "schilit_leverage_trend", "schilit_serial_acquirer", "goodwill_pct",
                 "dorsey_cfo_ni_divergence", "dorsey_ar_growth_flag", "lynch_inventory_flag",
                 "greenblatt_sector_excluded", "greenblatt_low_pe_flag", "mulford_fcf_consistent"]
    for col in flag_cols:
        data[col] = None

    sector = data.get("sector", "")
    pe = _sf(data.get("pe"))

    # Greenblatt exclusions (Greenblatt-only, not universe-wide)
    data["greenblatt_sector_excluded"] = bool(sector in ("Financial Services", "Utilities"))
    data["greenblatt_low_pe_flag"] = bool(pe is not None and 0 < pe < 5)

    if bs is None or bs.empty:
        return

    bs_cols = sorted(bs.columns)
    latest_bs = bs_cols[-1]

    # Goodwill %
    total_assets = _bs_row(bs, ["Total Assets"], latest_bs)
    gw = _bs_row(bs, ["Goodwill"], latest_bs)
    if gw is not None and total_assets and total_assets > 0:
        data["goodwill_pct"] = round(gw / total_assets * 100, 2)
    else:
        data["goodwill_pct"] = 0.0

    # A/R and Inventory analysis (need at least 2 years)
    if len(bs_cols) >= 2:
        prev_bs = bs_cols[-2]
        ar_now = _bs_row(bs, ["Accounts Receivable", "Net Receivables"], latest_bs)
        ar_prev = _bs_row(bs, ["Accounts Receivable", "Net Receivables"], prev_bs)
        inv_now = _bs_row(bs, ["Inventory"], latest_bs)
        inv_prev = _bs_row(bs, ["Inventory"], prev_bs)
        rev_y0 = _sf(data.get("revenue_y0"))
        rev_y1 = _sf(data.get("revenue_y1"))

        # S1: DSO
        if ar_now and rev_y0 and rev_y0 > 0:
            data["schilit_dso"] = round(ar_now / (rev_y0 / 365), 1)

        # S2: A/R vs Revenue divergence
        ar_growth = _pct_growth(ar_now, ar_prev)
        rev_growth = _pct_growth(rev_y0, rev_y1)
        if ar_growth is not None and rev_growth is not None:
            data["schilit_ar_revenue_divergence"] = round(ar_growth - rev_growth, 2)
            data["dorsey_ar_growth_flag"] = bool(ar_growth > rev_growth + 5)

        # S4: DSI
        if inv_now and income_stmt is not None and not income_stmt.empty:
            is_cols = sorted(income_stmt.columns)
            cogs = _bs_row(income_stmt, ["Cost Of Revenue", "Cost Of Goods Sold"], is_cols[-1])
            if cogs and cogs > 0:
                data["schilit_dsi"] = round(inv_now / (cogs / 365), 1)

        # S5: Inventory vs Revenue divergence
        inv_growth = _pct_growth(inv_now, inv_prev)
        if inv_growth is not None and rev_growth is not None:
            data["schilit_inventory_revenue_div"] = round(inv_growth - rev_growth, 2)
            data["lynch_inventory_flag"] = bool(inv_growth > rev_growth + 5)

    # S3: CapEx / Depreciation ratio
    if cashflow is not None and not cashflow.empty and income_stmt is not None and not income_stmt.empty:
        cf_cols = sorted(cashflow.columns)
        is_cols = sorted(income_stmt.columns)
        capex = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], cf_cols[-1])
        da = _bs_row(cashflow, ["Depreciation And Amortization", "Depreciation Amortization Depletion"], cf_cols[-1])
        if da is None:
            da = _bs_row(income_stmt, ["Reconciled Depreciation", "Depreciation And Amortization In Income Statement"], is_cols[-1])
        if capex is not None and da is not None and da != 0:
            data["schilit_capex_depr_ratio"] = round(abs(capex) / abs(da), 2)

    # S9: Working Capital as % of CFO
    if cashflow is not None and not cashflow.empty:
        cf_cols = sorted(cashflow.columns)
        wc_change = _bs_row(cashflow, ["Change In Working Capital", "Changes In Working Capital"], cf_cols[-1])
        ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], cf_cols[-1])
        if wc_change is not None and ocf is not None and ocf != 0:
            data["schilit_wc_cffo_pct"] = round(wc_change / ocf * 100, 2)

    # S11: Leverage Trend (debt growth - equity growth)
    if len(bs_cols) >= 2:
        debt_now = _bs_row(bs, ["Total Debt"], latest_bs)
        debt_prev = _bs_row(bs, ["Total Debt"], bs_cols[-2])
        eq_now = _bs_row(bs, ["Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest"], latest_bs)
        eq_prev = _bs_row(bs, ["Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest"], bs_cols[-2])
        debt_g = _pct_growth(debt_now, debt_prev)
        eq_g = _pct_growth(eq_now, eq_prev)
        if debt_g is not None and eq_g is not None:
            data["schilit_leverage_trend"] = round(debt_g - eq_g, 2)

    # S12: Serial Acquirer
    if len(bs_cols) >= 2:
        gw_prev = _bs_row(bs, ["Goodwill"], bs_cols[-2], 0)
        gw_now = gw or 0
        rev_growth = _pct_growth(_sf(data.get("revenue_y0")), _sf(data.get("revenue_y1")))
        gw_growth = _pct_growth(gw_now, gw_prev) if gw_prev and gw_prev > 0 else None
        if gw_growth is not None and rev_growth is not None:
            data["schilit_serial_acquirer"] = bool(gw_growth > 20 and rev_growth < 10)

    # R1 Dorsey: CFO/NI Divergence
    ni = [_sf(data.get(f"net_income_y{i}")) for i in range(3)]
    cfo_ni = _sf(data.get("schilit_cfo_ni_ratio"))
    if ni[0] is not None and ni[1] is not None and ni[0] > ni[1] and cfo_ni is not None and cfo_ni < 0.8:
        data["dorsey_cfo_ni_divergence"] = True
    else:
        data["dorsey_cfo_ni_divergence"] = False

    # M8: FCF Consistent
    if cashflow is not None and not cashflow.empty:
        cf_cols = sorted(cashflow.columns)
        fcf_positive_count = 0
        total_years = 0
        for col in cf_cols[-4:]:
            ocf = _bs_row(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"], col)
            capex = _bs_row(cashflow, ["Capital Expenditure", "Purchase Of PPE"], col)
            if ocf is not None and capex is not None:
                total_years += 1
                if (ocf + capex) > 0:
                    fcf_positive_count += 1
        data["mulford_fcf_consistent"] = bool(total_years >= 3 and fcf_positive_count == total_years)


# ─── 9. STOCK CLASSIFICATION (4 columns) ───

def compute_classification(data):
    """Lynch category (via the W2 archetype engine), debt health, lifecycle,
    and enterprising financial pass."""
    ni_cagr = _sf(data.get("ni_cagr_3y"))
    de = _sf(data.get("de"))

    # ── Lynch Category — W2: derived from the archetype engine ──
    # The old priority-ordered ladder keyed `cyclical` off graham_eps_cv > 0.5,
    # which split at the MEDIAN of the distribution it was splitting:
    # pct(eps_cv > 0.5) ran 0.321-0.558 in EVERY sector, and the ordering was
    # INVERTED (Technology 0.481 > Basic Materials 0.369). Cyclical share by
    # sector spanned only 24.1%-34.9% across the whole economy. It was a coin
    # flip, and it was load-bearing on lynch_score -> lynch_frac -> score.
    # Turnaround held top precedence and fired on any rounding-error loss, so it
    # also stole 660 rows from every other bucket. See archetype.py for the full
    # diagnostic record and the estimability principle behind the replacement.
    #
    # ONE classifier, TWO consumers: lynch_score branches on lynch_category
    # exactly as before — it simply stops receiving noise. Building a second
    # classifier alongside the broken one would have let the two drift.
    _arch = assign_archetype(data)
    data["archetype_primary"] = _arch["archetype_primary"]
    data["archetype_secondary"] = _arch["archetype_secondary"]
    data["archetype_confidence"] = _arch["archetype_confidence"]
    data["archetype_basis"] = _arch["archetype_basis"]
    data["lynch_category"] = lynch_category_from(_arch["archetype_primary"])

    # Lynch Debt Health
    if de is None:
        data["lynch_debt_healthy"] = "unknown"
    elif de <= 33:
        data["lynch_debt_healthy"] = "normal"
    elif de <= 100:
        data["lynch_debt_healthy"] = "acceptable"
    else:
        data["lynch_debt_healthy"] = "risky"

    # Mulford Lifecycle Stage
    ni_y0 = _sf(data.get("net_income_y0"))
    ocf_ni_ratio = _sf(data.get("schilit_cfo_ni_ratio"))
    consistent_cfo = data.get("dorsey_consistent_cfo")

    if ni_y0 is not None and ni_y0 < 0 and not consistent_cfo:
        data["mulford_lifecycle_stage"] = "startup"
    elif ni_y0 is not None and ni_y0 > 0 and ocf_ni_ratio is not None and ocf_ni_ratio < 0.5:
        data["mulford_lifecycle_stage"] = "growth"
    elif ni_y0 is not None and ni_y0 > 0 and ocf_ni_ratio is not None and ocf_ni_ratio >= 1.0:
        data["mulford_lifecycle_stage"] = "mature"
    elif ni_cagr is not None and ni_cagr < -5:
        data["mulford_lifecycle_stage"] = "decline"
    else:
        data["mulford_lifecycle_stage"] = "growth"  # default

    # E1: Enterprising Financial Pass
    cr = _sf(data.get("current_ratio"))
    data["graham_ent_financial_pass"] = bool(cr and cr >= 1.5 and de is not None and de <= 110)


# ─── 10. SPECTRUM SCORES (9 columns) ───

def _ramp(x, lo, hi):
    """Linear ramp, higher-is-better: 0 at/below lo, 1 at/above hi, clipped to [0,1].
    None/NaN -> 0.0 (fail-closed, matching Graham's conservative adaptation)."""
    x = _sf(x)
    if x is None:
        return 0.0
    if hi == lo:
        return 1.0 if x >= hi else 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def _ramp_down(x, ideal, limit):
    """Linear ramp, lower-is-better: 1 at/below ideal, 0 at/above limit, clipped.
    None/NaN -> 0.0 (fail-closed)."""
    x = _sf(x)
    if x is None:
        return 0.0
    if limit == ideal:
        return 1.0 if x <= ideal else 0.0
    return max(0.0, min(1.0, (limit - x) / (limit - ideal)))

# ══ Framework term tables — SINGLE SOURCE for every framework sub-score ══
# Each term returns a 3-tuple:  (integer_points, graded_points, input_columns)
#
#   integer_points  sum to the stored *_score column. compute_framework_verdicts
#                   thresholds THAT into the pass flags, so the integer half is
#                   what fires a drift alert.
#   graded_points   sum to *_graded / *_frac, which feed score_continuous.
#   input_columns   what the term actually reads.
#
# Both halves live in ONE table so a term can never be added to one and
# forgotten in the other. The comment at the Lynch block promised exactly that
# ("adds to BOTH on the same line so they cannot diverge"); this makes it
# structural instead of a convention.
#
# input_columns exists so the drift labeller can attribute a flip to the INPUT
# that moved, rather than assigning each term a standing price/business label.
# Every ratio term below has a price denominator and a fundamental numerator
# (div yield, PEG, PE, NCAV, one-dollar test, Graham number), so a standing
# label would be a proxy for the thing we actually want measured — the same
# substitution that made `pe`/`roe_pct` attribution collapse.
#
# VERIFIED against 4,473 stored v8 rows, 2026-07-31 (term_table_probe.py):
# both halves and all four pass flags reproduce exactly, all six Lynch
# categories, including 2,322 abstentions reproducing as None.
#
# Bump TERMS_VERSION whenever a term is added, removed or renamed. An
# entry_trace carrying an older version has a SHORTER vector; that must read as
# "not comparable", never as a term scoring zero.
TERMS_VERSION = 1
 
 
def terms_graham(data):
    """Graham Defensive, /8. Integer = book thresholds; graded = book ramps."""
    eps_growth = _sf(data.get("graham_eps_growth_pct_4y"))
    pe_pb = _sf(data.get("graham_pe_pb_composite"))
    gn = _sf(data.get("graham_number"))
    price = _sf(data.get("price"))
    gn_margin = ((gn - price) / price * 100) if (gn and price and price > 0) else None
    size = 1 if data.get("graham_adequate_size") else 0
    cr = 1 if data.get("graham_current_ratio_pass") else 0
    ltd = 1 if data.get("graham_ltd_vs_nca") else 0
    stable = 1 if data.get("graham_earnings_stable_4y") else 0
    div = 1 if (data.get("dividend_consecutive_years") or 0) >= 5 else 0
    epsg = 1 if (eps_growth is not None and eps_growth >= 33) else 0
    pepb = 1 if (pe_pb is not None and 0 < pe_pb <= 22.5) else 0
    gnum = 1 if (gn and price and price <= gn) else 0
    return {
        "D1_adequate_size":    (size, float(size), ("graham_adequate_size",)),
        "D2_current_ratio":    (cr, _ramp(data.get("current_ratio"), 1.5, 2.0),
                                ("graham_current_ratio_pass", "current_ratio")),
        "D3_ltd_vs_nca":       (ltd, float(ltd), ("graham_ltd_vs_nca",)),
        "D4_earnings_stable":  (stable, float(stable), ("graham_earnings_stable_4y",)),
        "D5_dividend_record":  (div, float(div), ("dividend_consecutive_years",)),
        "D6_eps_growth":       (epsg, _ramp(eps_growth, 0, 33),
                                ("graham_eps_growth_pct_4y",)),
        "D7c_pe_pb":           (pepb, float(pepb), ("graham_pe_pb_composite",)),
        "D8_graham_number":    (gnum, _ramp(gn_margin, 0, 30),
                                ("graham_number", "price")),
    }
 
 
def terms_dorsey(data):
    """Dorsey moat + Buffett tenets, /10. No percentile grading — book ramps and
    booleans only, so the framework stays absolute rather than relative."""
    fcf_m = _sf(data.get("dorsey_fcf_margin"))
    pm = _sf(data.get("profit_margin"))
    roa = _sf(data.get("dorsey_roa"))
    droic = _sf(data.get("dorsey_roic"))
    odt = _sf(data.get("buffett_one_dollar_test"))
    fcf_i = 1 if (fcf_m is not None and fcf_m >= 5) else 0
    pm_i = 1 if (pm is not None and pm >= 0.15) else 0
    roe_c = 1 if data.get("dorsey_roe_consistent") else 0
    roa_i = 1 if (roa is not None and roa >= 6) else 0
    roic_i = 1 if (droic is not None and droic > COST_OF_CAPITAL_PROXY) else 0
    unlev = 1 if data.get("buffett_roe_unleveraged") else 0
    odt_i = 1 if (odt is not None and odt >= 1.0) else 0
    alloc = 1 if data.get("buffett_rational_allocation") else 0
    ops = 1 if (data.get("graham_earnings_stable_4y")
                and data.get("dorsey_consistent_cfo")) else 0
    vcg = 1 if data.get("buffett_value_creating_growth") else 0
    return {
        "D1_fcf_margin":       (fcf_i, _ramp(fcf_m, 0, 5), ("dorsey_fcf_margin",)),
        "D2_profit_margin":    (pm_i, _ramp(pm, 0, 0.15), ("profit_margin",)),
        "D3_roe_consistent":   (roe_c, float(roe_c), ("dorsey_roe_consistent",)),
        "D4_roa":              (roa_i, _ramp(roa, 6, 7), ("dorsey_roa",)),
        "D5_roic_vs_wacc":     (roic_i, float(roic_i), ("dorsey_roic",)),
        "B1_roe_unleveraged":  (unlev, float(unlev), ("buffett_roe_unleveraged",)),
        "B2_one_dollar_test":  (odt_i, float(odt_i), ("buffett_one_dollar_test",)),
        "B3_rational_alloc":   (alloc, float(alloc), ("buffett_rational_allocation",)),
        "B4_consistent_ops":   (ops, float(ops),
                                ("graham_earnings_stable_4y", "dorsey_consistent_cfo")),
        "B5_value_creating":   (vcg, float(vcg), ("buffett_value_creating_growth",)),
    }
 
 
def terms_lynch(data):
    """Lynch, /10, BRANCHED on lynch_category. Each category is a DIFFERENT term
    set, so a category change replaces the whole vector — it must be reported as
    a reclassification, never as terms moving. `unknown` returns {} (abstention:
    no terms, not zero terms)."""
    cat = data.get("lynch_category", "unknown")
    peg = _sf(data.get("lynch_peg"))
    peg_adj = _sf(data.get("lynch_peg_adjusted"))
    debt_h = data.get("lynch_debt_healthy")
    pe = _sf(data.get("pe"))
    stable = 2 if data.get("graham_earnings_stable_4y") else 0
    cfo = 2 if data.get("dorsey_consistent_cfo") else 0
 
    if cat == "fast_grower":
        pi = 3 if (peg is not None and peg < 1) else \
             2 if (peg is not None and peg < 1.5) else \
             1 if (peg is not None and peg < 2) else 0
        gf = data.get("lynch_growth_flag")
        gi = 3 if gf == "ideal" else 2 if gf == "acceptable" else 0
        di = 2 if debt_h == "normal" else 1 if debt_h == "acceptable" else 0
        return {
            "FG1_peg":             (pi, 3 * _ramp_down(peg, 1.0, 2.0), ("lynch_peg",)),
            "FG2_growth_flag":     (gi, float(gi), ("lynch_growth_flag",)),
            "FG3_debt":            (di, float(di), ("lynch_debt_healthy",)),
            "FG4_earnings_stable": (stable, float(stable), ("graham_earnings_stable_4y",)),
        }
    if cat == "stalwart":
        pi = 3 if (pe is not None and 0 < pe <= 15) else \
             2 if (pe is not None and 0 < pe <= 20) else 0
        ai = 3 if (peg_adj is not None and peg_adj >= 2) else \
             2 if (peg_adj is not None and peg_adj >= 1) else 0
        di = 2 if debt_h in ("normal", "acceptable") else 0
        return {
            "ST1_pe":              (pi, (3 * _ramp_down(pe, 15, 20))
                                    if (pe is not None and pe > 0) else 0.0, ("pe",)),
            "ST2_peg_adjusted":    (ai, 3 * _ramp(peg_adj, 1, 2), ("lynch_peg_adjusted",)),
            "ST3_debt":            (di, float(di), ("lynch_debt_healthy",)),
            "ST4_earnings_stable": (stable, float(stable), ("graham_earnings_stable_4y",)),
        }
    if cat == "slow_grower":
        consec = data.get("dividend_consecutive_years", 0) or 0
        ci = 3 if consec >= 10 else 2 if consec >= 5 else 0
        payout = _sf(data.get("graham_payout_ratio"))
        pi = 3 if (payout and 30 <= payout <= 75) else \
             1 if (payout and payout < 30) else 0
        dy = _sf(data.get("dividend_yield"))
        yi = 2 if (dy and dy > 0.04) else 1 if (dy and dy > 0.02) else 0
        cb = 2 if data.get("dorsey_clean_balance_sheet") else 0
        return {
            "SG1_div_years":       (ci, float(ci), ("dividend_consecutive_years",)),
            "SG2_payout":          (pi, float(pi), ("graham_payout_ratio",)),
            "SG3_div_yield":       (yi, 2 * _ramp(dy, 0.02, 0.04), ("dividend_yield",)),
            "SG4_clean_bs":        (cb, float(cb), ("dorsey_clean_balance_sheet",)),
        }
    if cat == "cyclical":
        # Low PE = expensive (peak), high PE = cheap (trough), so PE is NOT
        # rewarded here. Every term is sign/categorical — nothing ramps.
        di = 3 if debt_h == "normal" else 2 if debt_h == "acceptable" else 0
        accel = _sf(data.get("lynch_growth_acceleration"))
        ai = 3 if (accel is not None and accel > 0) else 0
        iv = 2 if data.get("lynch_inventory_flag") is False else 0
        return {
            "CY1_debt":            (di, float(di), ("lynch_debt_healthy",)),
            "CY2_accel":           (ai, float(ai), ("lynch_growth_acceleration",)),
            "CY3_inventory":       (iv, float(iv), ("lynch_inventory_flag",)),
            "CY4_consistent_cfo":  (cfo, float(cfo), ("dorsey_consistent_cfo",)),
        }
    if cat == "turnaround":
        ncps = _sf(data.get("lynch_net_cash_per_share"))
        nc = 3 if (ncps is not None and ncps > 0) else 0
        di = 3 if debt_h == "normal" else 2 if debt_h == "acceptable" else 0
        ni_y0 = _sf(data.get("net_income_y0"))
        pf = 2 if (ni_y0 and ni_y0 > 0) else 0
        return {
            "TA1_net_cash":        (nc, float(nc), ("lynch_net_cash_per_share",)),
            "TA2_debt":            (di, float(di), ("lynch_debt_healthy",)),
            "TA3_now_profitable":  (pf, float(pf), ("net_income_y0",)),
            "TA4_consistent_cfo":  (cfo, float(cfo), ("dorsey_consistent_cfo",)),
        }
    if cat == "asset_play":
        ncav_r = _sf(data.get("graham_ncav_ratio"))
        ni = 4 if (ncav_r is not None and ncav_r <= 0.67) else \
             2 if (ncav_r is not None and ncav_r <= 1.0) else 0
        di = 3 if debt_h == "normal" else 0
        gnc = _sf(data.get("graham_net_cash"))
        nc = 3 if (gnc and gnc > 0) else 0
        return {
            "AP1_ncav":            (ni, (4 * _ramp_down(ncav_r, 0.67, 1.0))
                                    if (ncav_r is not None and ncav_r > 0) else 0.0,
                                    ("graham_ncav_ratio",)),
            "AP2_debt":            (di, float(di), ("lynch_debt_healthy",)),
            "AP3_net_cash":        (nc, float(nc), ("graham_net_cash",)),
        }
    return {}
 
 
def terms_trajectory(data):
    """Trajectory, /10. Direction of travel, price-independent — verified: not
    one input column below is a price or a price derivative."""
    ni_cagr = _sf(data.get("ni_cagr_3y"))
    rev_cagr = _sf(data.get("revenue_cagr_3y"))
    rev_g = _sf(data.get("rev_growth"))
    debt_g = _sf(data.get("debt_growth"))
    de = _sf(data.get("de"))
 
    t1i = 3 if (ni_cagr is not None and ni_cagr > 15) else \
          2 if (ni_cagr is not None and ni_cagr > 0) else 0
    t1g = (2 + _ramp(ni_cagr, 0, 15)) if (ni_cagr is not None and ni_cagr > 0) else 0.0
    t2 = 2 if (rev_cagr is not None and rev_cagr > 0) else 0
 
    m_now = m_then = None
    r0, n0 = _sf(data.get("revenue_y0")), _sf(data.get("net_income_y0"))
    if r0 and r0 > 0 and n0 is not None:
        m_now = n0 / r0
    for _back in (3, 2):                      # y3, falling back to y2
        rb = _sf(data.get(f"revenue_y{_back}"))
        nb = _sf(data.get(f"net_income_y{_back}"))
        if rb and rb > 0 and nb is not None:
            m_then = nb / rb
            break
    t3 = 2 if (m_now is not None and m_then is not None and m_now > m_then) else 0
 
    if debt_g is not None and rev_g is not None and debt_g < rev_g:
        t4 = 2
    elif debt_g is not None and debt_g < 0:
        t4 = 2
    else:
        t4 = 0
    t5 = 1 if (de is not None and de < 100) else 0
    return {
        "T1_earnings_cagr":    (t1i, t1g, ("ni_cagr_3y",)),
        "T2_revenue_cagr":     (t2, float(t2), ("revenue_cagr_3y",)),
        "T3_margin_expansion": (t3, float(t3),
                                ("revenue_y0", "net_income_y0", "revenue_y2",
                                 "net_income_y2", "revenue_y3", "net_income_y3")),
        "T4_debt_vs_growth":   (t4, float(t4), ("debt_growth", "rev_growth")),
        "T5_leverage_sane":    (t5, float(t5), ("de",)),
    }

# ── Components of every ratio input, for drift attribution ────────────────
# A term input like `pe` is a RATIO. When it moves, "price moved" and "earnings
# moved" are different events with different meanings for an owner, and the
# ratio alone cannot tell them apart. Naming the components lets the drift
# labeller measure which side moved on THIS row, instead of assigning the
# metric a standing price/business category — which would assert that the
# denominator usually wins, unmeasured.
#
# `price` is the price component. Everything else is fundamental. There is no
# list of "price-bearing metrics" anywhere; there is only the price column.
TERM_INPUT_COMPONENTS = {
    "pe":                ("price", "eps"),
    "lynch_peg":         ("price", "eps", "ni_cagr_3y"),
    "lynch_peg_adjusted": ("price", "eps", "ni_cagr_3y", "dividend_yield"),
    "graham_ncav_ratio": ("price", "graham_ncav_per_share"),
    "dividend_yield":    ("price", "eps", "graham_payout_ratio"),
}
 
# Ratio inputs whose components are NOT all stored. These get a delta and NO
# verdict: "Dorsey lost 1 point, one-dollar test 1.04 -> 0.91" is the whole
# truth available, and it beats a guessed category. Removing an entry here
# means storing the missing component, never substituting a near-enough one.
UNATTRIBUTABLE_INPUTS = frozenset({
    # = pe_on_AVERAGE_eps * pb. The 4-year average EPS is not a stored column,
    # and using trailing `eps` in its place is exactly the proxy substitution
    # this design exists to remove.
    "graham_pe_pb_composite",
    # = 4-year market-cap delta over retained earnings. Retained earnings is
    # not stored.
    "buffett_one_dollar_test",
})
 
_LYNCH_CATEGORIES = ("fast_grower", "stalwart", "slow_grower", "cyclical",
                     "turnaround", "asset_play")
 
 
def term_input_columns():
    """Every column a trace must carry to re-derive terms and attribute a flip.
 
    DERIVED from the term tables, never declared. A hand-written list silently
    goes short the first time a term gains an input — and a short trace reads
    as "the input held", which is the failure mode the whole design targets.
    Lynch is probed once per category because its branches are disjoint.
    """
    cols = set()
    for fn in (terms_graham, terms_dorsey, terms_trajectory):
        for t in fn({}).values():
            cols.update(t[2])
    for cat in _LYNCH_CATEGORIES:
        for t in terms_lynch({"lynch_category": cat}).values():
            cols.update(t[2])
    for c in list(cols):
        cols.update(TERM_INPUT_COMPONENTS.get(c, ()))
    # lynch_category selects the branch; greenblatt has no terms, only a
    # universe percentile, so it is attributed from its own two components.
    cols.update(("lynch_category", "greenblatt_earnings_yield", "greenblatt_roic"))
    return tuple(sorted(cols))
 
 
def framework_terms(data):
    """Every framework's term table for one row. The drift labeller's only
    input. Greenblatt is absent by design: it has no terms, only a universe
    percentile, so its flips are attributed from its own stored components."""
    return {"graham": terms_graham(data), "dorsey_buffett": terms_dorsey(data),
            "lynch": terms_lynch(data), "trajectory": terms_trajectory(data)}
 
 
def _sum_terms(terms):
    """(integer total, graded total) for one framework's term table.
 
    Deliberately an explicit left-to-right loop, NOT sum(). CPython >= 3.12
    gives sum() compensated (Neumaier) summation over floats, which is more
    accurate than the `+=` accumulation this replaces — and therefore NOT
    equal to it. On one row in 6,000 that moved dorsey_frac 0.4062 -> 0.4061
    and score_continuous with it, i.e. the refactor's output would have
    depended on the interpreter version. A behaviour-preserving refactor must
    reproduce the old arithmetic exactly, including its rounding error."""
    i_tot, g_tot = 0, 0.0
    for t in terms.values():
        i_tot += t[0]
        g_tot += t[1]
    return i_tot, g_tot


def compute_spectrum_scores(data):
    """Compute all spectrum scores from the layer-1 metrics."""

    # ── Graham Defensive Score (X/8) and GRADED (X.xx/8) ──
    # Both halves from ONE term table (terms_graham). Integer spine unchanged:
    # graham_pass, `score`, selection and backtest cohorts still read
    # graham_defensive_score.
    graham_d, graham_dg = _sum_terms(terms_graham(data))
    data["graham_defensive_score"] = graham_d
    data["graham_defensive_graded"] = round(graham_dg, 4)
    data["graham_frac"] = round(graham_dg / 8.0, 4)

    # ── Graham Enterprising Score (X/6) ──
    graham_e = 0
    if data.get("graham_ent_financial_pass"): graham_e += 1
    if data.get("graham_earnings_stable_4y"): graham_e += 1  # same check, relaxed = same with 4Y
    if data.get("graham_ent_has_dividend"): graham_e += 1
    if data.get("graham_ent_earnings_growing"): graham_e += 1
    ntav = _sf(data.get("graham_price_to_ntav"))
    if ntav is not None and 0 < ntav <= 1.2: graham_e += 1
    pe = _sf(data.get("pe"))
    if pe is not None and 0 < pe <= 10: graham_e += 1
    data["graham_enterprising_score"] = graham_e

    # ── Greenblatt Score (percentile bucket 1-10) ──
    # NOTE: This requires universe-level ranking. Set placeholder here.
    # Actual ranking computed in score_all_stocks() after all stocks processed.
    data["greenblatt_score"] = None  # Filled by rank pass
    data["greenblatt_frac"] = None   # Filled by rank pass (continuous percentile, 0-1)

    # ── Dorsey+Buffett Combined Score (X/10) and GRADED (X.xx/10) ──
    # Both halves from ONE term table (terms_dorsey). No percentile grading:
    # Dorsey is the pure-fundamental framework and must stay absolute.
    db_score, db_graded = _sum_terms(terms_dorsey(data))
    data["dorsey_buffett_score"] = db_score
    data["dorsey_buffett_graded"] = round(db_graded, 4)
    data["dorsey_frac"] = round(db_graded / 10.0, 4)
    fcf_m = _sf(data.get("dorsey_fcf_margin"))   # the 10-minute score below reads it

    # ── Dorsey 10-Minute Score (X/8) ──
    t10 = 0
    if data.get("dorsey_has_operating_profit"): t10 += 1
    if data.get("dorsey_consistent_cfo"): t10 += 1
    # ROE > 10% in 3/4 years
    roe_vals = [_sf(data.get(f"roe_y{i}")) for i in range(4)]
    valid_roe = [v for v in roe_vals if v is not None]
    if len(valid_roe) >= 3 and sum(1 for v in valid_roe if v >= 10) >= 3: t10 += 1
    # Earnings consistency (low CV)
    cv = _sf(data.get("graham_eps_cv"))
    if cv is not None and cv < 0.5: t10 += 1
    if data.get("dorsey_clean_balance_sheet"): t10 += 1
    if fcf_m is not None and fcf_m >= 5: t10 += 1
    # Low share dilution
    dil = _sf(data.get("dorsey_share_dilution_pct"))
    if dil is not None and dil < 2: t10 += 1
    elif dil is None: t10 += 1  # No data = benefit of doubt
    # FCF consistent
    if data.get("mulford_fcf_consistent"): t10 += 1
    data["dorsey_10min_score"] = t10

    # ── Lynch Score (category-branching) and GRADED ──
    # Both halves from ONE term table (terms_lynch). Each category is a
    # different term set; `unknown` returns {} — abstention, summing to 0
    # exactly as before, which compute_framework_verdicts turns into
    # lynch_pass = None rather than False.
    l_score, l_graded = _sum_terms(terms_lynch(data))
    data["lynch_score"] = l_score
    data["lynch_graded"] = round(l_graded, 4)
    data["lynch_frac"] = round(l_graded / 10.0, 4)

    # ── Schilit Manipulation Score (X/10, INVERTED: higher = worse) ──
    manip = 0
    ar_div = _sf(data.get("schilit_ar_revenue_divergence"))
    if ar_div is not None and ar_div > 10: manip += 1
    inv_div = _sf(data.get("schilit_inventory_revenue_div"))
    if inv_div is not None and inv_div > 10: manip += 1
    accruals = _sf(data.get("schilit_accruals_ratio"))
    if accruals is not None and accruals > 0.10: manip += 1
    cfo_ni = _sf(data.get("schilit_cfo_ni_ratio"))
    if cfo_ni is not None and cfo_ni < 0.5: manip += 1
    fcf_ni = _sf(data.get("schilit_fcf_ni_ratio"))
    if fcf_ni is not None and fcf_ni < 0: manip += 1
    capex_d = _sf(data.get("schilit_capex_depr_ratio"))
    if capex_d is not None and capex_d > 2.0: manip += 1
    wc_pct = _sf(data.get("schilit_wc_cffo_pct"))
    if wc_pct is not None and abs(wc_pct) > 50: manip += 1
    lev_trend = _sf(data.get("schilit_leverage_trend"))
    if lev_trend is not None and lev_trend > 20: manip += 1
    if data.get("schilit_serial_acquirer"): manip += 1
    if data.get("dorsey_cfo_ni_divergence"): manip += 1
    data["schilit_manipulation_score"] = manip

    # ── Mulford Cash Flow Quality Score (X/5) ──
    mq = 0
    ecm = _sf(data.get("mulford_ecm"))
    if ecm is not None and ecm > 0: mq += 1
    ecm_trend = _sf(data.get("mulford_ecm_trend"))
    if ecm_trend is not None and ecm_trend >= 0: mq += 1
    cm = _sf(data.get("mulford_cash_margin"))
    if cm is not None and cm > 5: mq += 1
    ocf_oi = _sf(data.get("mulford_ocf_oi_ratio"))
    if ocf_oi is not None and ocf_oi >= 1.0: mq += 1
    if data.get("mulford_fcf_consistent"): mq += 1
    data["mulford_cashflow_quality_score"] = mq


# ─── 11. QUALITY GATE (rewritten) ───

def compute_trajectory_score(data):
    """Trajectory (0-10): direction of travel over three years, price-independent.

    The other four frameworks all ask a question about price or structure:
      Graham     — is it cheap and safe?
      Greenblatt — is it cheap with high returns on capital?
      Dorsey     — does it have a moat?
      Lynch      — is its growth fairly priced?
    None asks whether the business is getting better. That is Trajectory's job,
    and its job alone.

    Design notes:
      - 3-year CAGRs, not 1-year deltas. A single year is noise.
      - Revenue AND earnings, not either. Revenue without profit is not progress.
      - Debt growth is measured AGAINST revenue growth, not against a constant.
        Leverage is only a problem when it outpaces the business it funds.
      - Missing inputs score zero. A metric we cannot measure is not a metric
        the company passed. This is the only honest treatment of NaN.
    """
    # Both halves from ONE term table (terms_trajectory). Clip preserved: the
    # term maxima sum to exactly 10, so it binds only defensively.
    t, t_graded = _sum_terms(terms_trajectory(data))
    data["trajectory_score"] = max(0, min(10, t))
    t_graded = max(0.0, min(10.0, t_graded))
    data["trajectory_graded"] = round(t_graded, 4)
    data["trajectory_frac"] = round(t_graded / 10.0, 4)


def compute_quality_gate(data):
    """Rewritten quality gate using manipulation flags + ECM."""
    quality = True

    accruals = _sf(data.get("schilit_accruals_ratio"))
    if accruals is not None and accruals > 0.10:
        quality = False

    manip_score = data.get("schilit_manipulation_score", 0) or 0
    if manip_score > 3:
        quality = False

    ecm_trend = _sf(data.get("mulford_ecm_trend"))
    if ecm_trend is not None and ecm_trend < -5.0:
        quality = False

    # Retain existing checks
    spike = _sf(data.get("earnings_spike"))
    if spike is not None and spike > 3.0:
        quality = False

    non_op = _sf(data.get("non_op_pct"))
    if non_op is not None and non_op > 40:
        quality = False

    data["quality_pass"] = quality


# ─── 12. FRAMEWORK PASS VERDICTS (5 dimensions) ───

def compute_framework_verdicts(data):
    """Compute the 5-framework pass/fail and 0-5 composite score."""
    graham_pass = (data.get("graham_defensive_score", 0) or 0) >= 6
    greenblatt_pass = (data.get("greenblatt_score") or 0) >= 7
    dorsey_buff_pass = (data.get("dorsey_buffett_score", 0) or 0) >= 6

    # Trajectory — now a 0-10 sub-score, thresholded like the other four.
    # Old rule was `(rev>0 OR ni>0) AND (debt shrank OR de<50)`, which fired on
    # 39.6% of the universe: the AND of a tautology (72.4%) and a coin (52.4%).
    # Threshold 6 is a CALIBRATION KNOB — tune against the measured base rate
    # after the rerun so Trajectory sits alongside Dorsey (12.7%) and Lynch
    # (23.9%) rather than dominating the composite.
    # Calibrated 2026-07 against measured base rates on 4,461 stocks:
    #   >=6 -> 40.5%   >=7 -> 33.0%   >=8 -> 20.0%   >=9 -> 12.4%
    # 8 sits between dorsey (14.6%) and lynch (23.0%). At 8 the 4+ pool holds
    # 100 stocks, 23 Large, large-cap quota ceiling 20 (defensive needs 8).
    # 9 is unaffordable: 5/5 count halves to ten.
    #
    # PROVISIONAL. Two known issues, both for Sprint 12 to settle empirically:
    #   1. 578 stocks sit at exactly 7 — this boundary is a cliff, not a slope.
    #   2. The `de < 100` component is a LEVEL test inside a DIRECTION framework,
    #      and Graham/Dorsey already test leverage. It systematically penalizes
    #      capital-intensive businesses (NTPC: ni_cagr 17%, margins expanding,
    #      scores 7). Reconstruct the alternative with `trajectory_score - (de<100)`
    #      and compare forward returns before deciding.
    trajectory_pass = (data.get("trajectory_score", 0) or 0) >= 8

    # Lynch pass (category-dependent threshold)
    lynch_s = data.get("lynch_score", 0) or 0
    cat = data.get("lynch_category", "unknown")
    # Threshold: ≥6 out of 10 for most categories
    lynch_threshold = 6
    if cat == "slow_grower":
        lynch_threshold = 5  # Easier to pass for dividend payers
    # W2 abstention. `unclassified` -> lynch_category "unknown", which matches
    # no branch at 1235-1305, so lynch_s stays 0 and `0 >= 6` recorded a
    # FAILURE on 2,320 rows (51.8% of universe). They have not failed Lynch;
    # Lynch cannot evaluate them. selector._applicable_frameworks:379 already
    # excludes them from the GATE -- this fixes the stored CLAIM, which is the
    # half that was never wired. archetype.py:232-234 documents it as pending.
    lynch_pass = None if cat == "unknown" else (lynch_s >= lynch_threshold)

    data["graham_pass"] = graham_pass
    data["greenblatt_pass"] = greenblatt_pass
    data["dorsey_pass"] = dorsey_buff_pass  # Keep column name for backward compat
    data["trajectory_pass"] = trajectory_pass
    data["lynch_pass"] = lynch_pass
    # `is True` keeps score BYTE-IDENTICAL to pre-abstention behaviour: None
    # and False both contribute 0. No archive breaks, no reconcile break, no
    # disturbance to backtest cohorts. The denominator is NOT stored here --
    # score_history.applicable (jsonb, written from _applicable_frameworks) is
    # the single source of truth for it. A second denominator computed in a
    # second module is the pattern this project rejects.
    _verdicts = (graham_pass, greenblatt_pass, dorsey_buff_pass,
                 trajectory_pass, lynch_pass)
    data["score"] = sum(1 for _v in _verdicts if _v is True)
    # Continuous mirror of the integer score (0-5). None frac -> 0, mirroring the
    # integer (where an unranked greenblatt fails its pass). Display/W1 only.
    data["score_continuous"] = round(sum(
        (_sf(data.get(f)) or 0)
        for f in ("graham_frac", "greenblatt_frac", "dorsey_frac", "lynch_frac", "trajectory_frac")
    ), 4)


# ═══════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════

def compute_all_deep_metrics(data, stock):
    """
    Main entry point. Computes all 90 new columns.

    Args:
        data: dict with basic fundamentals (from fetch_fundamentals)
        stock: yfinance Ticker object (for dividend history, balance sheet detail)
    """
    info = stock.info or {}
    shares = _sf(info.get("sharesOutstanding"))

    # Fetch raw data from yfinance
    try:
        income_stmt = stock.financials
        if income_stmt is None or income_stmt.empty:
            income_stmt = None
    except Exception:
        income_stmt = None

    try:
        balance_sheet = stock.balance_sheet
        if balance_sheet is None or balance_sheet.empty:
            balance_sheet = None
    except Exception:
        balance_sheet = None

    try:
        cashflow = stock.cashflow
        if cashflow is None or cashflow.empty:
            cashflow = None
    except Exception:
        cashflow = None

    # Compute all categories
    compute_balance_sheet(data, info, balance_sheet, shares)
    compute_earnings_quality(data, info, income_stmt, cashflow, balance_sheet, shares)
    compute_moat(data, info, income_stmt, balance_sheet, cashflow, shares)
    compute_valuation(data, info, income_stmt, cashflow, balance_sheet, shares)
    compute_growth(data, income_stmt, shares)
    compute_dividends(data, stock, income_stmt, shares)
    compute_management(data, info, income_stmt, cashflow, balance_sheet, shares, stock)
    compute_manipulation_flags(data, income_stmt, balance_sheet, cashflow)
    compute_classification(data)
    compute_spectrum_scores(data)
    compute_trajectory_score(data)
    compute_quality_gate(data)
    compute_axis_scores(data)

    return data


def compute_greenblatt_ranks(all_data):
    """
    Universe-level computation: rank all stocks on ROIC and Earnings Yield,
    compute combined rank and percentile score.
    Must be called AFTER all stocks have individual metrics computed.
    """
    # Filter to non-excluded stocks with valid ROIC and EY
    rankable = []
    for d in all_data:
        if d.get("greenblatt_sector_excluded"):
            continue
        roic = _sf(d.get("greenblatt_roic"))
        ey = _sf(d.get("greenblatt_earnings_yield"))
        if roic is not None and ey is not None:
            rankable.append(d)

    if not rankable:
        return

    # Sort by ROIC (descending = best gets rank 1)
    rankable.sort(key=lambda x: (-(_sf(x.get("greenblatt_roic")) or 0), str(x.get("ticker"))))
    for i, d in enumerate(rankable):
        d["_roic_rank"] = i + 1

    # Sort by Earnings Yield (descending = best gets rank 1)
    rankable.sort(key=lambda x: (-(_sf(x.get("greenblatt_earnings_yield")) or 0), str(x.get("ticker"))))
    for i, d in enumerate(rankable):
        d["_ey_rank"] = i + 1

    # Combined rank
    for d in rankable:
        d["greenblatt_combined_rank"] = d.get("_roic_rank", 0) + d.get("_ey_rank", 0)

    # Convert to percentile score (1-10)
    rankable.sort(key=lambda x: (x.get("greenblatt_combined_rank", 9999), str(x.get("ticker"))))
    n = len(rankable)
    for i, d in enumerate(rankable):
        percentile = (1 - i / n) * 10
        d["greenblatt_score"] = max(1, min(10, int(round(percentile))))
        d["greenblatt_frac"] = round(percentile / 10.0, 4)   # continuous, un-bucketed rank position

    # Now compute framework verdicts for ALL stocks (including greenblatt)
    for d in all_data:
        compute_framework_verdicts(d)
