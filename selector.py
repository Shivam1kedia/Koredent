"""
selector.py — deterministic portfolio construction.

PURE. No Streamlit. No network. No LLM. Takes data in, returns picks out.
That purity is not hygiene: it is what lets backtest_runner.py call the REAL
selector with a point-in-time price slice instead of reimplementing it and
backtesting a lookalike.

    select_portfolio(universe_df, policy, price_history) -> SelectionResult

WHAT THIS REPLACES, AND WHY
---------------------------
The old get_sip_candidates ended with:

    candidates.sort(key=lambda c: c.get("diversification_rank", 999))

Every ranking computed above that line was discarded. diversification_rank came
from a greedy minimum-variance loop over a covariance matrix estimated from one
year of daily closes. For a thinly traded stock, days with no trade produce a
flat close, hence a ZERO return, hence a downward-biased sigma; and
non-synchronous trading biases every correlation downward too (Scholes-Williams).
So argmin(portfolio variance) is mechanically argmax(staleness). The loop was a
staleness detector wearing a Markowitz costume, and it was choosing the stocks.

That is why every portfolio was penny stocks. It is also why every portfolio was
IDENTICAL regardless of the questionnaire: diversification_rank contains no user
information at all.

THE HIERARCHY, RESTORED
----------------------
  Tier 1  investability floor    — existence, not preference
  Tier 2  the gate               — Q8, hard, abstention-adjusted
  Tier 3  the ranking            — Q7 x Q9, continuous, WITHIN SECTOR
  Struct  integer quotas         — resolved up front, reservation not repair
  Covar   staleness + tiebreak   — never security selection

Reilly & Brown builds the efficient frontier FROM the assets you are willing to
hold. It never asks covariance to choose them.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
 
# Drift attribution re-derives terms from stored inputs on BOTH sides, so the
# term list exists once, in the file that scores. deep_metrics imports only
# math/pandas/datetime/archetype/os/json and nothing in that chain imports
# selector, so this edge is cheap and acyclic.
from deep_metrics import (framework_terms, term_input_columns,
                          TERM_INPUT_COMPONENTS, UNATTRIBUTABLE_INPUTS,
                          TERMS_VERSION)
 
# DERIVED from the term tables at import, never declared here. A hand-kept
# list goes short the first time a term gains an input, and a short trace
# reads as "the input held" - the exact failure this design removes.
TERM_INPUT_COLS = term_input_columns()

# ── Tier 1: investability floor ────────────────────────────────────────────
# Calibrated 2026-07 against the measured distribution of the 4+ pool.
# ₹500cr x ₹50L keeps 100 of 113. It removes, among others:
#   505685.BO  ₹7cr    ₹0.0 lakh/day   (name field corrupted)
#   Comfort Fincap  ₹54cr   ₹3.6 lakh/day
#   PATINTLOG.NS    ₹110cr  ₹26 lakh/day   score 4   <- junk reaches 4/5
MIN_MARKET_CAP = 500e7          # ₹500 crore. Equivalently: risk_tier != "Micro".
MIN_TURNOVER = 50e5             # ₹50 lakh/day = price x avg_daily_volume.

# Turnover is NOT an exit-liquidity gate — a ₹5,000 SIP holding ₹333 per name
# could exit anything. It is a PRICE INTEGRITY gate. A stock trading ₹10 lakh a
# day has a few hundred prints, prices stale by construction, and a tape an
# operator can move. That is what corrupts the covariance matrix and what makes
# the cheapness flags fire on prices nobody transacted at.
#
# Which makes turnover a coarse prefilter and staleness the precise instrument
# for the same underlying property. Hence both, at modest thresholds.
MIN_NONZERO_RETURN_FRAC = 0.90
MIN_RETURN_OBSERVATIONS = 120

# ── Struct: the ruin floor ─────────────────────────────────────────────────
# NOT Evans & Archer. Their 12-18 was measured on RANDOMLY SELECTED portfolios,
# where stock #15 has the same expected return as stock #1 and diversification is
# free. Under a ranking, stock #15 is your fifteenth-best idea and costs expected
# return. Importing their number into a selected portfolio is a category error.
#
# This floor is a RUIN constraint, not a variance constraint, and it is derived
# rather than invented: generate_ips caps any single stock at 10% (SEBI). Equal
# weight is 100/n, which exceeds 10% whenever n < 10. Below n=10 the IPS
# literally contradicts itself.
MIN_STOCKS_RUIN_FLOOR = 10

FRAMEWORKS = ("graham", "greenblatt", "dorsey_buffett", "trajectory", "lynch")

# The boolean gates and the continuous sub-scores they threshold.
# Four of the five booleans are literally `subscore >= k` (compute_framework_verdicts).
# So the sub-score IS the native continuous underlying — no proxy, no judgement call.
# Five booleans give 32 orderings across 4,461 stocks. Five 0-10 sub-scores give ~10^4.
# THAT is what lets the questionnaire answers move the portfolio at all.
PASS_FLAG = {
    "graham": "graham_pass",
    "greenblatt": "greenblatt_pass",
    "dorsey_buffett": "dorsey_pass",
    "trajectory": "trajectory_pass",
    "lynch": "lynch_pass",
}
SUBSCORE = {
    "graham": "graham_defensive_score",
    "greenblatt": "greenblatt_score",
    "dorsey_buffett": "dorsey_buffett_score",
    "trajectory": "trajectory_score",
    "lynch": "lynch_score",
}
# The graded decimals (W0.1). Same checks, but continuous: the fractional part is
# distance-to-threshold on the book-ramped checks. The integer versions above are
# step functions — within one sector dozens of stocks share a value and collapse to
# the SAME percentile, which is dead weight the demand tilt cannot move. Ranking
# only; the GATE still counts the PASS_FLAG booleans, so `score` is unaffected.
SUBSCORE_GRADED = {
    "graham": "graham_defensive_graded",
    "greenblatt": "greenblatt_frac",
    "dorsey_buffett": "dorsey_buffett_graded",
    "trajectory": "trajectory_graded",
    "lynch": "lynch_graded",
}

# W1. The NORMALISED (0-1) graded decimals. Distinct from SUBSCORE_GRADED above,
# which holds the RAW graded values on their native scales (Graham /8, the other
# four /10). Selection ranks on PERCENTILES of the raw values, so scale never
# matters there. Drift decomposition compares magnitudes ACROSS frameworks
# ("which one moved most"), where mixing a /8 with a /10 would silently favour
# the /10s. score_continuous is the sum of exactly these five, so
# delta(score_continuous) decomposes into delta(fracs) with no residual.
FRAC_COL = {
    "graham": "graham_frac",
    "greenblatt": "greenblatt_frac",
    "dorsey_buffett": "dorsey_frac",
    "trajectory": "trajectory_frac",
    "lynch": "lynch_frac",
}

# Q9. The user names what they VALUE, not what they will WAIVE. So it tilts the
# ranking; it never opens a side door in the gate. A 5/5 stock with the best
# trajectory in its sector must win through the front door, not be displaced by
# a 4/5 admitted through an exception.
TRADEOFF_TILT = {
    "ok_fail_graham": ("graham",),
    "ok_fail_trajectory_lynch": ("trajectory", "lynch"),
    "ok_fail_dorsey_buffett": ("dorsey_buffett",),
    "any": (),
}

# Each framework is a BUNDLE of axes — the composition derived while building the
# axis table. This is what lets ONE demand tilt (Q4/Q6/Q7/Q9 -> Quality/Growth/
# Price/Safety) drive framework weights, instead of Q7/Q9 tilting the frameworks
# AND the axes separately — the same preference counted twice.
# Weighting a framework up for one axis used to smuggle in its other axes: a user
# who wants "cheap" got Graham boosted, which also carries Safety and Quality.
FRAMEWORK_AXIS_COMPOSITION = {
    "graham":         {"price": 0.5, "safety": 0.3, "quality": 0.2},
    "greenblatt":     {"price": 0.5, "quality": 0.5},
    "dorsey_buffett": {"quality": 0.8, "safety": 0.2},
    "trajectory":     {"growth": 0.8, "quality": 0.2},
    "lynch":          {"growth": 0.5, "price": 0.5},
}

# Q7. Continuous tiebreak when two stocks tie on the weighted sub-score rank.
# (column, higher_is_better)
PHILOSOPHY_TIEBREAK = {
    "deep_value": ("graham_margin_of_safety_pct", True),
    "contrarian": ("pct_from_low", False),
    "quality_compounder": ("dorsey_roic", True),
    # lynch_peg_adjusted = (ni_cagr_3y + dividend_yield_pct) / pe, higher better.
    # This column was garbage until Sprint 11.4: dividend_yield arrived from
    # yfinance as a percent, was multiplied by 100 again, and the dividend term
    # ran 3.4x the growth term at the median. The ingest fix repaired it.
    "growth_at_fair_price": ("lynch_peg_adjusted", True),
}

# 2026-07-28: swept {0.005, 0.01, 0.02, 0.03, 0.05} against thresholds
# pre-registered before the numbers were seen (median population <= 2, p90 <= 4).
# At 0.05 the free pool held a median of 6 candidates and p90 of 10 -- not a tie
# SET but a slate, and reordering ten candidates by covariance is security
# selection, which this band explicitly disclaims. 0.02 is the largest width
# that stays a tie set (median 2, p90 3-4) while leaving the tiebreak active on
# 45-70% of calls. Caveat: 11-20 calls per site, so the 0.02/0.03 boundary rests
# on a thin sample. band_sweep.py reproduces.
RANK_BAND = 0.02

# ── The conviction sleeve ──────────────────────────────────────────────────
# Measured 2026-07: gate 3+ and gate 2+ produced IDENTICAL portfolios under all
# four philosophies. Widening the gate admits 557 stocks and none can reach the
# top fifteen of their sector, because _rank_score = Σ wᵢ·pctᵢ and the dominant
# weight is capped at 35%:
#
#   specialist (99th on Graham, 40th elsewhere)  0.35(0.99) + 0.65(0.40) = 0.607
#   generalist (70th on everything)              1.00(0.70)              = 0.700
#
# The specialist loses. A gate is a floor; a floor removes, it never surfaces.
# Middle-tier value has to come from the RANKING, and the ranking punishes
# specialization by construction.
#
# So: reserve k slots for the top-decile specialist under the user's dominant
# framework, gate-exempt but Tier-1 bound, and make the trace say so out loud:
#   "Fails three of our five frameworks. 99th percentile of Graham value in its
#    sector. You told us that's what you're hunting."
#
# PROVISIONAL. This encodes an unmeasured belief — that one framework's signal
# justifies holding a stock that fails three others. Sprint 12 measures forward
# return per flag and sets k from data. graham_pass is rarest (5.6%) and most
# orthogonal (max |phi| = 0.26), so it is the one most likely to carry alpha.
# Gate-respecting, so safe to enable at 3+: a specialist who clears the user's
# own bar is precisely who this is for. Left at 0 for 4+ — "only the best"
# already means broad agreement, and a buried 4/5 specialist is a rounding error.
CONVICTION_SLOTS = {4: 0, 3: 1, 2: 2}
CONVICTION_MIN_PCT = 0.90

# W2 sector-shrinkage strength: pct = w*pct_sector + (1-w)*pct_pool, w = n/(n+K).
# SPECIFICATION, not fitted. A percentile's resolution is ~1/n, so below roughly
# 20 observations a within-group percentile carries less information than the
# parent distribution. K = 20 puts the crossover there.
#
# Interaction with CONVICTION_MIN_PCT, stated because it is NOT obvious: that
# threshold is on the percentile VALUE, and shrinkage compresses small-sector
# percentiles. Top-of-sector alone clears 0.90 only once n >= ~180. Below that,
# a conviction pick must ALSO stand up pool-wide:
#     n=9  -> needs pool pct >= 0.855      n=59  -> needs >= 0.605
#     n=27 -> needs pool pct >= 0.765      n=100 -> needs >= 0.400
# Intended. "Best of 9" is thinner evidence than "best of 400", and a
# concentrated conviction bet is precisely where that distinction should bite.
SECTOR_SHRINK_K = 20


# ══════════════════════════════════════════════════════════════════════════
# TIER 1 — INVESTABILITY FLOOR
# ══════════════════════════════════════════════════════════════════════════
def _tier1(df: pd.DataFrame, sip_amount: float, rejects: dict) -> pd.DataFrame:
    """Existence, not preference. A stock failing these is un-exitable,
    un-priceable, or un-assessable — not merely low quality."""
    n0 = len(df)

    def cut(mask, reason):
        nonlocal df
        removed = int((~mask).sum())
        if removed:
            rejects[reason] = rejects.get(reason, 0) + removed
        df = df[mask]

    cut(df["quality_pass"] != False, "quality_gate")            # noqa: E712
    cut(df["years_of_data"] >= 2, "insufficient_history")
    cut(df["pe"].notna() & (df["pe"] > 0), "loss_making_or_no_pe")

    # NOT notna(roe_pct) / notna(de). Those were vestigial: once we rank on
    # sub-scores, nothing downstream reads either column, and they cost us
    # CRISIL (roe_pct NaN) and every lender (de NaN). Missing data already has
    # an honest path — it lowers the sub-score, which fails the boolean, which
    # lowers `score`, which the gate rejects. A blanket notna() DOUBLE-COUNTS
    # that penalty, converting "we could not measure it" into "excluded".

    # Structural: no sector => cannot be checked against max_same_sector or
    # min_sectors. Do not let these become an "Unknown" sector that spuriously
    # satisfies the breadth requirement.
    # W2 — DECLARED, not incidental. This is not a rare edge case: 2,177 of
    # 4,477 fresh rows (48.6%) have no sector, and the boundary is an EXCHANGE
    # boundary, not a coverage gradient — yfinance classifies 2,300 of 2,387 NSE
    # tickers and ZERO BSE-only listings (0 classified / 2,090 blank).
    #
    # Recovering it was considered and REJECTED. BSE's own INDUSTRY field is
    # already parsed in universe_updater's listing fetch and dropped at the
    # combine step, so the data is reachable — it is just not worth reaching
    # for. Of those 2,177 rows, exactly NINE score >= 4 and ZERO score 5; 72.5%
    # score 0 against 35.7% for classified rows; median market cap is ₹44 Cr
    # with p75 at ₹130 Cr, i.e. three-quarters sit below the ₹200 Cr
    # adequate-size floor. They score 0 because their cashflow statements are
    # thin and _ramp fails closed on None — recovering the LABEL does not
    # recover the FUNDAMENTALS. Worse, adding ~2,000 data-poor names to sector
    # denominators would inflate every classified stock's percentile without
    # adding information. Revisit only if BSE fundamental coverage improves.
    cut(df["sector"].notna(), "no_sector")

    # A row whose `name` is blank or a comma-mangled fragment is a corrupt CSV
    # record, not a company. The universe has held `505685.BO,0P0000CFCT,0` and
    # a broken TRANSRAILL row. Never render one to a user.
    _nm = df["name"].astype(str).str.strip()
    cut(_nm.str.len() > 2, "corrupt_name")

    if "is_unevaluable" in df.columns:
        # Declared, not accidental. Lenders stayed out before only because
        # `notna(de)` happened to drop them — banks report no debtToEquity.
        cut(~df["is_unevaluable"].fillna(False).astype(bool), "unevaluable_business_model")

    if "is_stale" in df.columns:
        # Carry-forward rows: a stock throttled by Yahoo today, filled from its
        # last committed fundamentals so it doesn't vanish from monitoring. Fine
        # for the tracker to SEE (position continuity), never eligible for a NEW
        # buy — allocating fresh capital on stale numbers is the one thing
        # carry-forward must not enable. The asymmetry is deliberate: skipping a
        # good buy for a day costs a day; buying into a stale row is unrecoverable.
        # .get-guarded so pre-Sprint-14 CSVs (no column) are unaffected.
        cut(~df["is_stale"].fillna(False).astype(bool), "stale_carryforward")

    cut(df["market_cap"] >= MIN_MARKET_CAP, "below_market_cap_floor")

    _turnover = df["price"] * df["avg_daily_volume"]
    cut(_turnover.fillna(0) >= MIN_TURNOVER, "below_turnover_floor")

    # Affordability. The user's rule, unchanged: one share must fit inside one
    # SIP installment. The old code silently DROPPED this filter when fewer
    # than 10 stocks survived, handing back a portfolio the user cannot buy.
    # Fail loudly instead.
    cut(df["price"] <= float(sip_amount), "unaffordable_at_sip")

    rejects["_tier1_in"] = n0
    rejects["_tier1_out"] = len(df)
    return df


def _staleness_filter(df: pd.DataFrame, price_history: pd.DataFrame | None,
                      rejects: dict) -> pd.DataFrame:
    """A stock whose price does not move on 10%+ of trading days is UNPRICEABLE
    for covariance. Chapter 6's variance algebra presupposes continuously priced
    securities; feed it stale prices and it ranks tickers by deadness.

    This runs BEFORE any covariance is computed, so the staleness attractor never
    gets a chance to operate."""
    if price_history is None or price_history.empty:
        return df

    cols = [t for t in df["ticker"] if t in price_history.columns]
    if not cols:
        return df

    # fill_method=None pins pandas 3.0's semantics. Under pandas 2.x the default
    # was 'pad', which forward-filled a stale price into a ZERO return — exactly
    # the artefact this filter exists to catch.
    rets = price_history[cols].pct_change(fill_method=None)

    obs = rets.count()
    nonzero = (rets.fillna(0) != 0).sum()
    frac = (nonzero / obs.replace(0, np.nan)).fillna(0)

    ok = set(obs[(obs >= MIN_RETURN_OBSERVATIONS) & (frac >= MIN_NONZERO_RETURN_FRAC)].index)
    # Tickers absent from price_history are not penalised; we simply have no
    # basis to judge them and the covariance step will skip them.
    absent = set(df["ticker"]) - set(cols)
    keep = df["ticker"].isin(ok | absent)

    removed = int((~keep).sum())
    if removed:
        rejects["stale_prices"] = removed
    return df[keep]


# ══════════════════════════════════════════════════════════════════════════
# TIER 2 — THE GATE (Q8)
# ══════════════════════════════════════════════════════════════════════════
def _applicable_frameworks(row) -> tuple:
    """Greenblatt's formula uses ROIC and earnings yield, which are meaningless
    for a levered balance sheet. He says so himself: do not apply it to
    financials or utilities. Scoring them as though they FAILED it is applying
    it to them.

    Result today: exactly ONE stock in Financial Services reaches 4/5, out of a
    universe where financials are a third of the index. Not a market fact — an
    arithmetic ceiling of 4/5, judged against a 4+ gate.

    W2 — the SAME principle now applies to LYNCH. His method is
    categorise-then-evaluate: fast grower, stalwart, slow grower, cyclical,
    turnaround, asset play. When the archetype engine cannot place a business in
    any of them, the method does not run — there is no branch to score it on.
    lynch_score has no else-branch, so an unclassified row silently took
    lynch_frac = 0.0 and lynch_pass = False, which records "FAILED Lynch". It has
    not failed Lynch; Lynch cannot evaluate it. Two different claims.

    This is genuine INAPPLICABILITY, not missing data. Missing data already has
    an honest path — it lowers the sub-score, which fails the boolean. An
    unclassifiable business is a different thing: the framework takes no position
    on it at all. Both exclusions can fire together, leaving 3 applicable;
    _effective_gate holds it to the same FRACTION, so a 4+ gate becomes 2 of 3."""
    excluded = set()
    if bool(row.get("greenblatt_sector_excluded", False)):
        excluded.add("greenblatt")
    # Same reason, different balance-sheet shape. EBIT / (nwc + ppe) with a
    # non-positive denominator is not a return on capital -- negative for a
    # profitable business, positive for a loss-making one. Greenblatt's book
    # is silent on it and his policy where the formula cannot price something
    # is to exclude, not impute. Missing-data cases (no ca/cl, no EBIT) are
    # deliberately NOT here: those keep failing.
    if bool(row.get("greenblatt_capital_nonpositive", False)):
        excluded.add("greenblatt")
    if str(row.get("lynch_category") or "").strip() == "unknown":
        excluded.add("lynch")
    if not excluded:
        return FRAMEWORKS
    return tuple(f for f in FRAMEWORKS if f not in excluded)


def _effective_gate(min_score: int, n_applicable: int) -> int:
    """Hold every stock to the same FRACTION of applicable tests.

    Subtracting one from the score works at a 4+ gate and quietly becomes a
    SUBSIDY at 2+: a bank would need 1 of 4 (25%) against everyone else's
    2 of 5 (40%). Proportional does not have that failure mode.

        80% of 4 tests = 3.2. You cannot pass 3.2 tests. Nearest whole = 3.

    int(x + 0.5) rather than round(), because round() is banker's rounding and
    round(2.5) == 2. Not reachable with these inputs, but not worth relying on.
    """
    if n_applicable >= len(FRAMEWORKS):
        return int(min_score)
    return max(1, int(min_score * n_applicable / len(FRAMEWORKS) + 0.5))


# ── The FORCED-EXIT ceiling ────────────────────────────────────────────────
# _effective_gate is a FLOOR (admit at >= k). This is a CEILING (force an exit
# at <= t). They are not the same operation and must not share a formula.
#
# The raw rule was `now_score == 1`, an integer on an implied 5-denominator:
# sell when at most one fifth of the frameworks pass. Written as a fraction that
# survives abstention, the largest integer s with s/n <= 1/5 is floor(n/5)
# EXACTLY — no rounding choice, no new parameter.
#
#     n=5 -> 1     n=4 -> 0     n=3 -> 0
#
# The old integer 1 was invariant across n, which reads reassuring and is not:
# it held a 3-framework stock to a 33% exit bar and a 4-framework stock to 25%,
# against 20% for everyone else. The threshold stood still while abstention
# walked scores down into it. Measured 2026-07-27: GREENPOWER.NS went 2-of-4 to
# 1-of-3 in a single universe run because lynch_category flipped to unknown.
# Trajectory unchanged, Graham unchanged, nothing about the business moved, and
# the next review would have said SELL ALL.
#
# One principle governs both directions — NEVER make the abstaining stock worse
# off than the 5-denominator rule. _effective_gate satisfies it by being
# lenient-or-equal to the exact form; this satisfies it by being exact.
#
# Note this also SUBSUMES the score-0 branch: at n < 5 the threshold is 0, so
# "no thesis at all" is the only forced exit. Score 0 sells at every n either
# way, which is correct — "every applicable framework fails" carries no
# denominator.
FORCED_EXIT_BASE = 1   # 1-of-5: the fraction the raw integer rule already meant


def forced_exit_threshold(n_applicable: int) -> int:
    """Largest score that still forces an exit, at this denominator."""
    n = int(n_applicable or len(FRAMEWORKS))
    return (FORCED_EXIT_BASE * n) // len(FRAMEWORKS)


def forced_exit_applies(universe_row_or_df, ticker: str, now_score: int) -> bool:
    """True when `now_score` is at or below the forced-exit ceiling for THIS
    stock's applicable-framework count.

    Missing row -> assume the full five, preserving today's behaviour. That path
    is unreachable from the review caller (a missing row yields now_score 0,
    which sells at every denominator), but a silent default that CHANGED
    behaviour would be the worse failure.
    """
    df = universe_row_or_df
    try:
        row = df[df["ticker"] == ticker]
        if len(row) == 0:
            return int(now_score) <= forced_exit_threshold(len(FRAMEWORKS))
        row = row.iloc[0]
    except Exception:
        return int(now_score) <= forced_exit_threshold(len(FRAMEWORKS))
    n = len(_applicable_frameworks(row))
    return int(now_score) <= forced_exit_threshold(n)


# ── Score thresholds against the TRUE denominator ─────────────────────────
# Every surface that gated on the raw integer `score` compared a numerator
# counted over APPLICABLE frameworks to a threshold derived from FIVE. A stock
# where Lynch abstains is scored out of 4 and held to 5/5's bar, so it silently
# drops out of opportunity emails, replacement lists and radar pools.
# Measured 2026-07-27 on 4,626 rows: 54% of the universe has n_applicable = 4.
#
# `score` and score_applicable were numerically identical on all 4,626 rows —
# an abstaining framework's pass flag is False and contributes 0 either way.
# These helpers still recount from PASS_FLAG rather than trusting that equality:
# it is an empirical fact about today's data, not a guarantee.
def score_applicable(df: pd.DataFrame):
    """(n_applicable, score_applicable) per row, from the single source of
    truth. Benchmarked at 0.40s on 4,626 x 125 — cheap enough that no second,
    vectorised copy of _applicable_frameworks needs to exist."""
    n, s = [], []
    for _, r in df.iterrows():
        app = _applicable_frameworks(r)
        n.append(len(app))
        s.append(sum(1 for f in app if bool(r.get(PASS_FLAG[f], False))))
    return pd.Series(n, index=df.index), pd.Series(s, index=df.index)


def meets_score_mask(df: pd.DataFrame, k: int) -> pd.Series:
    """Boolean mask: does each row clear a k-of-5 FLOOR at its own denominator.
    Drop-in for `df["score"] >= k`. Wraps _effective_gate — no new convention."""
    n, s = score_applicable(df)
    return s >= n.map(lambda x: _effective_gate(k, int(x)))


def score_tiers(df: pd.DataFrame) -> pd.Series:
    """Tier 4 / 3 / 2 / None as FRACTION BANDS. Drop-in for the `score == 4`,
    `== 3`, `== 2` equality partition.

    Fraction is forced here, not preferred. Under the integer convention at
    n = 4: tier 4 -> int(3.2+.5) = 3, tier 3 -> int(2.4+.5) = 2, tier 2 ->
    int(1.6+.5) = 2. Tiers 3 and 2 COLLIDE. Rounding survives one boundary, not
    a partition of several.

    Also fixes a defect the equality form carried independently of abstention:
    `score == 4` never matched `score == 5`, so perfect-score stocks were
    excluded from the candidate list entirely. Measured: 15 such rows."""
    n, s = score_applicable(df)
    out = []
    for _n, _s in zip(n, s):
        f = (_s / _n) if _n else 0.0
        out.append(4 if f >= 4 / 5 else 3 if f >= 3 / 5 else 2 if f >= 2 / 5 else None)
    return pd.Series(out, index=df.index, dtype=object)


# ── Score DROP across time, on a stable denominator ───────────────────────
# `entry_score - current_score >= 2` compares two integers counted over
# possibly DIFFERENT applicable sets. When a holding loses a framework to
# abstention, `score` falls by one with no business change and half the alert
# threshold is spent before anything happens. Latent until W2 made Lynch
# abstention live; measured 2026-07-27 on 21 holdings — LUPIN 5->4 and
# GREENPOWER 4->3 each carried exactly one artifact point.
#
# Rule: count only frameworks applicable at BOTH ends. A framework that left is
# counted at neither. No new parameter — the threshold is the 2-of-5 the raw
# rule already meant, written so it survives a variable denominator:
#     ceil(2 * n_common / 5)  ->  5:2  4:2  3:2  2:1
#
# Ratified fail direction: below MIN_COMPARABLE the score_drop alert does NOT
# fire. Two frameworks is the least on which "deteriorated by 40%" means
# anything, and a wrong LABEL is worse than silence — quality_pass and price
# alerts are independent and still fire. A trace without applicable/passed
# falls back to the RAW delta and is marked, so nothing silently stops
# alerting.
SCORE_DROP_BASE = 2        # 2-of-5: the fraction the raw integer rule meant
MIN_COMPARABLE = 2         # below this, "40% worse" is not a claim we can make


def score_drop_threshold(n_common: int) -> int:
    """Smallest integer drop meeting the 2-of-5 fraction at this denominator."""
    return max(1, math.ceil(SCORE_DROP_BASE * int(n_common) / len(FRAMEWORKS)))


def comparable_score_drop(entry_trace, universe_row, entry_score, current_score) -> dict:
    """Has the thesis deteriorated, counted only over comparable frameworks."""
    et = entry_trace or {}
    e_app, e_pass = et.get("applicable"), et.get("passed")
    if not isinstance(e_app, list) or not isinstance(e_pass, list):
        raw = int(entry_score) - int(current_score)
        return {"comparable": False, "n_common": None, "entry_common": None,
                "current_common": None, "delta": raw,
                "fires": raw >= SCORE_DROP_BASE}
    c_app = list(_applicable_frameworks(universe_row))
    c_pass = [f for f in c_app if bool(universe_row.get(PASS_FLAG[f], False))]
    common = [f for f in FRAMEWORKS if f in set(e_app) & set(c_app)]
    e = sum(1 for f in common if f in set(e_pass))
    c = sum(1 for f in common if f in set(c_pass))
    return {"comparable": True, "n_common": len(common), "entry_common": e,
            "current_common": c, "delta": e - c,
            "fires": len(common) >= MIN_COMPARABLE
                     and (e - c) >= score_drop_threshold(len(common))}


def score_drop_headline(name: str, cmp: dict, entry_score, current_score) -> str:
    """The raw pair is FALSE for a holding whose denominator moved — LUPIN would
    read '4 -> 3' when nothing changed. State the basis alongside the numbers."""
    if cmp.get("comparable"):
        return (f"{name} score dropped {cmp['entry_common']} -> "
                f"{cmp['current_common']} on the {cmp['n_common']} frameworks "
                f"comparable to entry")
    return f"{name} score dropped {entry_score} -> {current_score}"


def score_label(row, score=None) -> str:
    """Render a composite score with its TRUE denominator: "3 of 4", not "3/5".

    Every display surface must use this. Hardcoding "/5" states that a stock
    failed a test that was never applied to it — the same error _effective_gate
    and _applicable_frameworks exist to prevent, leaking back in at the last
    inch. A financial does not fail Greenblatt; an unclassifiable business does
    not fail Lynch.

    ONLY THE DENOMINATOR MOVES. The raw integer `score` is the sum of five
    *_pass booleans, and an abstaining framework's flag is already False, so it
    contributes 0 to the numerator either way. That is why this is a pure
    display fix and why nothing that WRITES a score has to change: the stored
    value stays the raw 5-denominator integer that portfolio_tracker compares
    against, exactly as documented at the watchlist insert site.

    row   : a universe row (dict or Series) carrying greenblatt_sector_excluded
            and lynch_category. A row missing them yields the full 5 — correct
            for pre-v4 archive rows, which were scored under v3 semantics and
            must not be retroactively reinterpreted.
    score : numerator override, for callers holding a stored score whose row is
            looked up separately. Defaults to row["score"].
    """
    if score is None:
        try:
            score = row.get("score")
        except AttributeError:
            score = None
    try:
        if score is None or pd.isna(score):
            return "—"
        score = int(score)
    except (TypeError, ValueError):
        return "—"
    n = len(_applicable_frameworks(row if row is not None else {}))
    return f"{score} of {n}"


def score_label(row, score=None) -> str:
    """Render a composite score with its TRUE denominator: "3 of 4", not "3/5".

    Every display surface must use this. Hardcoding "/5" states that a stock
    failed a test that was never applied to it — the same error _effective_gate
    and _applicable_frameworks exist to prevent, leaking back in at the last
    inch. A financial does not fail Greenblatt; an unclassifiable business does
    not fail Lynch.

    ONLY THE DENOMINATOR MOVES. The raw integer `score` is the sum of five
    *_pass booleans, and an abstaining framework's flag is already False, so it
    contributes 0 to the numerator either way. That is why this is a pure
    display fix and why nothing that WRITES a score has to change: the stored
    value stays the raw 5-denominator integer that portfolio_tracker compares
    against, exactly as documented at the watchlist insert site.

    row   : a universe row (dict or Series) carrying greenblatt_sector_excluded
            and lynch_category. A row missing them yields the full 5 — correct
            for pre-v4 archive rows, which were scored under v3 semantics and
            must not be retroactively reinterpreted.
    score : numerator override, for callers holding a stored score whose row is
            looked up separately. Defaults to row["score"].
    """
    if score is None:
        try:
            score = row.get("score")
        except AttributeError:
            score = None
    try:
        if score is None or pd.isna(score):
            return "—"
        score = int(score)
    except (TypeError, ValueError):
        return "—"
    n = len(_applicable_frameworks(row if row is not None else {}))
    return f"{score} of {n}"


def _tier2(df: pd.DataFrame, policy: dict, rejects: dict):
    """Annotate EVERY investable stock with its gate arithmetic, then return
    (gated_pool, annotated_frame). The conviction sleeve needs the arithmetic
    for stocks that FAILED the gate — that is the whole point of the sleeve.

    avoid_sectors moved to select_portfolio: a user's sector exclusion must bind
    on conviction candidates too, and _tier2 no longer sees them all.
    """
    min_score = int(policy.get("min_acceptable_score", 3))

    keep, applicable_col, score_col, gate_col = [], [], [], []
    for _, row in df.iterrows():
        app = row["_applicable"] if "_applicable" in row else _applicable_frameworks(row)
        s = sum(1 for f in app if bool(row.get(PASS_FLAG[f], False)))
        g = _effective_gate(min_score, len(app))
        keep.append(s >= g)
        applicable_col.append(app)
        score_col.append(s)
        gate_col.append(g)

    df = df.assign(_score_applicable=score_col, _effective_gate=gate_col)
    mask = pd.Series(keep, index=df.index)
    rejects["below_score_gate"] = int((~mask).sum())
    return df[mask], df


# ══════════════════════════════════════════════════════════════════════════
# TIER 3 — THE RANKING (Q7 x Q9), WITHIN SECTOR
# ══════════════════════════════════════════════════════════════════════════
def _resolve_weights(policy: dict) -> dict:
    """SINGLE tilt source. Q4/Q6/Q7/Q9 -> demand_tilt (per-axis lean) -> framework
    weights, via each framework's axis composition. Never touches the gate.

    Ranking still runs on the framework SUB-SCORES (sector-relative, abstention-aware,
    and carrying every boolean book check that the continuous-only axis scores
    structurally cannot). Only the WEIGHTS come from the demand tilt.

    Falls back to the legacy per-philosophy weights when a profile predates the tilt.
    """
    tilt = policy.get("demand_tilt") or {}
    if tilt:
        w = {f: sum(share * float(tilt.get(ax, 1.0))
                    for ax, share in FRAMEWORK_AXIS_COMPOSITION[f].items())
             for f in FRAMEWORKS}
    else:
        w = dict(policy.get("framework_weights") or {}) or {f: 20 for f in FRAMEWORKS}
        # legacy path only — Q9's halving now lives inside derive_demand_tilt
        for f in TRADEOFF_TILT.get(policy.get("acceptable_tradeoff", "any"), ()):
            w[f] = w.get(f, 0) / 2.0
    total = sum(w.values()) or 1.0
    return {f: 100.0 * w.get(f, 0) / total for f in FRAMEWORKS}


def _rank_population(df: pd.DataFrame, framework: str) -> pd.Series:
    """Which rows belong in `framework`'s RANKED POPULATION.

    Deliberately NOT _applicable_frameworks. That function answers the GATE's
    question — "did this stock pass a test it faced" — and is untouched by this
    change. This answers "does this row belong in the denominator of a
    percentile". Same inputs, different question. They stay separate functions
    because merging them is how an abstention concept acquires a second meaning,
    which is the failure mode W2 spent a sprint unpicking.
    """
    if framework == "greenblatt":
        excl = df.get("greenblatt_sector_excluded", pd.Series(False, index=df.index))
        return ~excl.fillna(False).astype(bool)
    if framework == "lynch":
        cat = df.get("lynch_category", pd.Series("", index=df.index))
        return cat.astype(str).str.strip() != "unknown"
    return pd.Series(True, index=df.index)


def _framework_percentile(df: pd.DataFrame, col: str, pop: pd.Series) -> pd.Series:
    """Sector/pool blended percentile computed over `pop` ONLY.

    Rows outside the population, and rows inside it with no value, get NaN and
    are weighted out of _rank_score entirely. Previously they sat in the
    denominator: lynch abstainers as a literal 0.0 at the bottom (inflating
    everyone else by +0.123 on that component) and greenblatt NaNs at the TOP,
    because na_option="bottom" places NaN at the HIGHEST percentiles — the
    opposite of what the name reads like, and of the .fillna(0.0) that used to
    sit beside it, which never fired.

    Measured before the change (n=661 post-tier1): the applicable-row mean
    matched closed form to three decimals — lynch (163+249.5)/661 = 0.624
    observed 0.624; greenblatt 310/661 = 0.469 observed 0.470 — and the spread
    scaled by exactly n_app/n. A framework's EFFECTIVE weight was its nominal
    weight x its applicable fraction. Lynch ran at ~75% of the weight
    _resolve_weights assigned it. Nobody chose that.

    w = n/(n+K) now counts the per-(sector, framework) population, since the
    population differs by framework. A sector with no members falls back
    entirely to the pool percentile at w = 0 — the direction W2 already argued
    for thin sectors.
    """
    vals = pd.to_numeric(df[col], errors="coerce")
    mask = pop & vals.notna()
    out = pd.Series(np.nan, index=df.index)
    if not mask.any():
        return out
    sub_vals = vals.loc[mask]
    sub_sector = df.loc[mask, "sector"]
    p_pool = sub_vals.rank(pct=True, method="average")
    p_sector = sub_vals.groupby(sub_sector).rank(pct=True, method="average")
    n_sector = sub_sector.groupby(sub_sector).transform("size")
    w_sector = (n_sector / (n_sector + SECTOR_SHRINK_K)).fillna(0.0)
    out.loc[mask] = w_sector * p_sector.fillna(0.0) + (1.0 - w_sector) * p_pool.fillna(0.0)
    return out


def _attach_applicable(df: pd.DataFrame) -> pd.DataFrame:
    """Which frameworks can even evaluate this stock. Needed by rank AND gate."""
    return df.assign(_applicable=[_applicable_frameworks(r) for _, r in df.iterrows()])


def _tier3(df: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Percentile-rank each sub-score WITHIN SECTOR, then weight.

    Within sector, because a 20% ROIC means something different in Utilities
    than in Software — and because a global rank would silently become a bet on
    whichever sector happens to be cheap this quarter. It also produces exactly
    the per-sector queues the quota filler consumes.

    Percentile rather than z-score: bounded, NaN-safe, and immune to the fat
    tails that riddle Indian small-cap fundamentals.

    W2 — SHRUNK toward the pool. A raw within-sector percentile scores "best of
    9" and "best of 400" identically at 1.00. In the post-tier1 pool the
    smallest sector runs to 9 members and the median to 59, so this is not
    hypothetical. Each sector percentile is blended with the pool-wide one:

        pct = w * pct_sector + (1 - w) * pct_pool,   w = n / (n + K)

    Order WITHIN a sector is unchanged: w is constant inside a sector and both
    percentiles are monotone in the metric, so the blend is too. What changes is
    the LEVEL — thin-sector stocks are pulled toward their pool standing, which
    is the entire point. Sector breadth is protected by min_sectors /
    max_same_sector in _tier2, not by inflated small-sector percentiles.
    """
    w = _resolve_weights(policy)
    df = _attach_applicable(df.copy())

    for f in FRAMEWORKS:
        col = SUBSCORE_GRADED.get(f)
        if col not in df.columns:          # pre-W0.1 archive rows
            col = SUBSCORE[f]
        if col not in df.columns:
            df[f"_pct_{f}"] = 0.0
            continue
        df[f"_pct_{f}"] = _framework_percentile(df, col, _rank_population(df, f))

    # Weight over frameworks whose PERCENTILE EXISTS, renormalized — not over
    # frameworks that apply. A stock that abstains from one is not penalised for
    # the absent fifth, and a stock we could not compute is not credited with a
    # percentile it never earned. The GATE still counts _applicable; these are
    # two different questions and _ranked_on records where they diverge.
    def _rank(row):
        app = row["_applicable"]
        live = [f for f in app if pd.notna(row[f"_pct_{f}"])]
        if not live:
            return 0.0
        wt = sum(w[f] for f in live) or 1.0
        return sum(w[f] * row[f"_pct_{f}"] for f in live) / wt

    df["_rank_score"] = df.apply(_rank, axis=1)
    df["_ranked_on"] = df.apply(
        lambda r: tuple(f for f in r["_applicable"] if pd.notna(r[f"_pct_{f}"])), axis=1)

    tb_col, tb_high = PHILOSOPHY_TIEBREAK.get(
        policy.get("philosophy", "growth_at_fair_price"), (None, True))
    if tb_col and tb_col in df.columns:
        v = pd.to_numeric(df[tb_col], errors="coerce")
        df["_tiebreak"] = (v if tb_high else -v).fillna(-np.inf)
        df["_tiebreak_metric"] = tb_col
        df["_tiebreak_value"] = v
    else:
        df["_tiebreak"] = 0.0
        df["_tiebreak_metric"] = None
        df["_tiebreak_value"] = np.nan

    # No _rank_in_sector / _sector_depth, no sort here. Those are TRACE fields —
    # "#1 of 14 in Technology" must count the stocks that actually competed for
    # the slot, i.e. the POST-gate pool. select_portfolio computes them after
    # _tier2. The _rank_score percentiles above are deliberately computed
    # PRE-gate, against everything investable.
    return df


# ══════════════════════════════════════════════════════════════════════════
# STRUCT — INTEGER QUOTAS, RESERVATION NOT REPAIR
# ══════════════════════════════════════════════════════════════════════════
def _affordable_n(prices: list[float], sip_amount: float) -> int:
    """Largest k such that the k cheapest names cost <= one SIP installment,
    one share each. This is what allocate_shares actually does breadth-first.

    generate_ips uses `sip_amount // 250`; get_sip_candidates used `// 500`.
    Same quantity, two numbers. Derive it from real prices instead."""
    total, k = 0.0, 0
    for p in sorted(prices):
        if total + p > sip_amount:
            break
        total += p
        k += 1
    return k


def _quotas(n: int, alloc: dict) -> dict:
    per_sector = max(1, min(int(alloc.get("max_same_sector", 3)),
                            int(alloc.get("max_sector_pct", 25) * n / 100)))
    return {
        "n": n,
        "large_min": math.ceil(alloc.get("large_cap_min_pct", 30) * n / 100),
        "mid_min": math.ceil(alloc.get("mid_cap_min_pct", 20) * n / 100),
        "small_max": math.floor(alloc.get("small_cap_max_pct", 25) * n / 100),
        "micro_max": 0,
        "per_sector": per_sector,
        "min_sectors": int(alloc.get("min_sectors", 3)),
    }


def _feasible(pool: pd.DataFrame, q: dict) -> tuple[bool, str]:
    """Check BEFORE filling, not by discovering mid-loop.

    The old code discovered infeasibility inside a greedy loop and repaired it
    with a swap that was free to evict your best stock for a worse large-cap."""
    if q["large_min"] + q["mid_min"] > q["n"]:
        return False, "large_min + mid_min exceeds n"
    for tier, need in (("Large", q["large_min"]), ("Mid", q["mid_min"])):
        by_sec = pool[pool["risk_tier"] == tier].groupby("sector").size()
        ceiling = sum(min(v, q["per_sector"]) for v in by_sec.values)
        if ceiling < need:
            return False, f"only {ceiling} {tier} reachable, need {need}"
    if pool["sector"].nunique() < q["min_sectors"]:
        return False, f"{pool['sector'].nunique()} sectors, need {q['min_sectors']}"
    return True, ""


def _corr_tiebreak(cands: pd.DataFrame, chosen: list, corr: pd.DataFrame | None):
    """Covariance's ONLY role in selection: reorder within a quality band.

    Take candidates whose _rank_score is within RANK_BAND of the best remaining,
    then prefer the one least correlated with what we already hold. It can never
    leapfrog a band. It never performs security selection.
    """
    if cands.empty:
        return None
    best = cands["_rank_score"].iloc[0]
    band = cands[cands["_rank_score"] >= best - RANK_BAND]
    if len(band) <= 1 or corr is None or not chosen:
        return cands.index[0]

    held = [c for c in chosen if c in corr.columns]
    if not held:
        return band.index[0]

    scores = {}
    for idx, row in band.iterrows():
        t = row["ticker"]
        scores[idx] = corr.loc[t, held].abs().mean() if t in corr.columns else 0.5
    return min(scores, key=scores.get)

def _mark_conviction(gated: pd.DataFrame, weights: dict) -> tuple[pd.DataFrame, str]:
    """Flag top-decile specialists WITHIN the gated pool.

    The first version of this drew from stocks that FAILED the gate. At a 2+
    gate, failing means score_applicable < 2, so every conviction pick was
    necessarily a 1/5 — we handed a user who asked for "2+ with a compelling
    reason" a stock passing one framework. The sleeve was structurally unable
    to surface 2/5 and 3/5 at all; it surfaced the bottom tier.

    The gate is the user's stated minimum and is inviolable. The problem was
    never the gate — it is that _rank_score BURIES specialists inside the pool.
    A 90th-percentile-Graham stock at 3/5 clears a 2+ gate easily and then loses
    every slot to broad 4/5 names. It is right there, unreachable.

    So the sleeve reorders within the gate instead of reaching beneath it.

    Three guards, each load-bearing:
      1. The dominant framework's BOOLEAN must pass. 99th percentile in a sector
         where nobody clears Graham is a fact about the sector, not the stock.
      2. The framework must APPLY (a utility cannot have Greenblatt conviction).
      3. Percentiles come from the PRE-gate frame, so "top decile" means top
         decile of everything investable, not of whatever survived the gate.
    """
    dom = max(weights, key=weights.get)
    g = gated.copy()
    if g.empty:
        g["_conviction_pct"] = np.nan
        g["_conviction_rank"] = np.nan
        g["_conviction_eligible"] = False
        return g, dom

    applies = g["_applicable"].apply(lambda a: dom in a)
    passes = g[PASS_FLAG[dom]].fillna(False).astype(bool)
    pct = g[f"_pct_{dom}"]

    g["_conviction_pct"] = pct.where(applies & passes)
    g["_conviction_eligible"] = applies & passes & (pct >= CONVICTION_MIN_PCT)

    # A conviction pick was chosen on the DOMINANT FRAMEWORK's percentile, not on
    # _rank_score. Reporting its composite rank is therefore misleading, and
    # occasionally unsayable: an early run surfaced BUILDPRO at greenblatt_pct
    # 0.936 and rank_in_sector #74 of 74. Both numbers were true. "Ranked last of
    # 74, and we bought it" is not a sentence any user accepts, however
    # well-founded. Give the trace the rank that actually did the choosing.
    g["_conviction_rank"] = (g.groupby("sector")[f"_pct_{dom}"]
                              .rank(ascending=False, method="first")
                              .where(applies & passes))
    return g, dom


def _fill(pool: pd.DataFrame, q: dict, corr, k_conviction: int = 0) -> tuple[list, dict]:
    """pool carries a boolean `_conviction` column. Conviction rows are invisible
    to the merit passes and are admitted only in their own pass, AFTER every IPS
    minimum is satisfied. They consume FREE slots, never quota slots."""
    # Conviction rows are NOT a separate population — they live in the same
    # gated pool and may well be taken on merit. The sleeve only fires for
    # specialists the merit passes left behind. That is the entire point.
    eligible = (pool["_conviction_eligible"] if "_conviction_eligible" in pool.columns
                else pd.Series(False, index=pool.index))
    chosen, tickers = [], []
    sec_count, tier_count = defaultdict(int), defaultdict(int)
    slot_type = {}

    def can_take(row) -> bool:
        t, s = row["risk_tier"], row["sector"]
        if sec_count[s] >= q["per_sector"]:
            return False
        if t == "Micro" and tier_count["Micro"] >= q["micro_max"]:
            return False
        if t == "Small" and tier_count["Small"] >= q["small_max"]:
            return False
        return True

    def take(idx, why):
        row = pool.loc[idx]
        chosen.append(idx)
        tickers.append(row["ticker"])
        sec_count[row["sector"]] += 1
        tier_count[row["risk_tier"]] += 1
        slot_type[idx] = why

    def remaining(mask=None):
        m = ~pool.index.isin(chosen)
        if mask is not None:
            m &= mask
        sub = pool[m]
        return sub[sub.apply(can_take, axis=1)] if len(sub) else sub

    # Pass 1: reserve the large-cap quota. Reservation, not post-hoc repair.
    while tier_count["Large"] < q["large_min"] and len(chosen) < q["n"]:
        c = remaining(pool["risk_tier"] == "Large")
        if c.empty:
            break
        take(_corr_tiebreak(c, tickers, corr), "cap_quota_large")

    # Pass 2: reserve the mid-cap quota.
    while tier_count["Mid"] < q["mid_min"] and len(chosen) < q["n"]:
        c = remaining(pool["risk_tier"] == "Mid")
        if c.empty:
            break
        take(_corr_tiebreak(c, tickers, corr), "cap_quota_mid")

    # Pass 3: breadth. One name per uncovered sector until min_sectors.
    while len(sec_count) < q["min_sectors"] and len(chosen) < q["n"]:
        c = remaining(~pool["sector"].isin(sec_count.keys()))
        if c.empty:
            break
        take(c.index[0], "breadth")

    # Pass 4: the conviction sleeve. AFTER every IPS minimum is met, so a
    # long-shot can never displace the large-cap floor or sector breadth.
    # Ordered by the dominant framework's percentile, not by _rank_score —
    # _rank_score is precisely the number that buried these stocks.
    taken_conv = 0
    while taken_conv < k_conviction and len(chosen) < q["n"]:
        c = remaining(eligible)
        if c.empty:
            break
        # Ordered by the dominant framework's percentile, NOT by _rank_score.
        # _rank_score is precisely the number that buried these stocks.
        take(c.sort_values("_conviction_pct", ascending=False).index[0], "conviction")
        taken_conv += 1

    # Pass 5: free fill, pure merit order.
    while len(chosen) < q["n"]:
        c = remaining()
        if c.empty:
            break
        take(_corr_tiebreak(c, tickers, corr), "free")

    return chosen, slot_type


def _jsonable(v):
    """Trace values that are NOT all numbers: term inputs include booleans and
    category strings (lynch_debt_healthy, lynch_growth_flag, lynch_category).
    _num would flatten every one of those to None. Numerics are NOT rounded
    here - these are re-scored by framework_terms(), not displayed, and
    rounding an input before a threshold test can flip the term it feeds.
    NaN -> None for the same reason as _num: NaN != NaN makes every later
    diff report a change that never happened.
    """
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, float, np.integer, np.floating)):
        f = float(v)
        return f if math.isfinite(f) else None
    return str(v)
 
 
def _num(v, nd: int = 4):
    """Trace numbers must be JSON-safe and COMPARABLE. NaN/None/inf -> None;
    anything else -> a rounded float.

    A NaN written into entry_trace is poison: NaN != NaN, so every later diff
    reports a change that never happened. None vs None compares equal, which is
    the honest answer for a metric we never had."""
    try:
        if v is None or pd.isna(v):
            return None
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


# ══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════
def select_portfolio(universe_df: pd.DataFrame, policy: dict,
                     price_history: pd.DataFrame | None = None) -> dict:
    """Deterministic. Same inputs, same portfolio. The LLM's only remaining job
    is to phrase the trace in plain English — it makes no structural decision."""
    rejects: dict = {}
    sip = float(policy.get("sip_amount", 5000))
    alloc = dict(policy.get("allocation_policy") or {})
    warnings: list[str] = []

    df = _tier1(universe_df.copy(), sip, rejects)
    df = _staleness_filter(df, price_history, rejects)

    # RANK FIRST, GATE SECOND.
    #
    # Percentiles must be computed against everything INVESTABLE, not against
    # the gated subset. Otherwise "top decile" means something different at a
    # 4+ gate (≈100 peers) than at 2+ (≈450), every survivor is renormalized
    # upward, and the gate silently cancels its own effect: a distinctive 2/5
    # Graham name loses exactly the distinctiveness that admitting it was
    # supposed to surface.
    #
    # A stock's standing is a property of the stock, not of the filter it
    # happened to pass through.
    # avoid_sectors binds on conviction candidates too, so it applies here —
    # before ranking, before the gate — not inside _tier2.
    avoid = set(policy.get("avoid_sectors") or [])
    if avoid:
        _n = len(df)
        df = df[~df["sector"].isin(avoid)]
        rejects["sector_excluded_by_user"] = _n - len(df)

    ranked = _tier3(df, policy)
    gated, annotated = _tier2(ranked, policy, rejects)

    weights = _resolve_weights(policy)
    k_conv = CONVICTION_SLOTS.get(int(policy.get("min_acceptable_score", 3)), 0)
    pool, dom_framework = _mark_conviction(gated, weights)
    if not k_conv:
        pool["_conviction_eligible"] = False
        dom_framework = None

    if pool.empty:
        return {"holdings": [], "warnings": ["No stock clears the gate you chose."],
                "rejects": rejects, "diagnostics": {}}

    # rank_in_sector / sector_depth are TRACE fields — "#1 of 14 in Technology"
    # must count the stocks that actually competed, i.e. post-gate.
    pool = pool.copy()
    pool["_rank_in_sector"] = (pool.groupby("sector")["_rank_score"]
                                   .rank(ascending=False, method="first").astype(int))
    pool["_sector_depth"] = pool.groupby("sector")["ticker"].transform("size")
    pool = pool.sort_values(["_rank_score", "_tiebreak"], ascending=False).reset_index(drop=True)

    corr = None
    if price_history is not None and not price_history.empty:
        cols = [t for t in pool["ticker"] if t in price_history.columns]
        if len(cols) >= 2:
            r = price_history[cols].pct_change(fill_method=None).dropna(how="all")
            # dropna(how="all"), not dropna(): a row-wise dropna across 200
            # tickers lets ONE gappy ticker delete that date for every other one.
            corr = r.corr(min_periods=MIN_RETURN_OBSERVATIONS)

    # ── n is endogenous. No padding, no truncation. ──
    ips_target = int((policy.get("portfolio_sizing") or {}).get("ips_target", 15))
    aff_n = _affordable_n(pool["price"].tolist(), sip)
    n = min(len(pool), aff_n, ips_target)

    if n < MIN_STOCKS_RUIN_FLOOR:
        warnings.append(
            f"Only {n} holdings are possible ({len(pool)} clear your "
            f"{policy.get('min_acceptable_score')}+ gate; {aff_n} fit one SIP "
            f"installment). Below {MIN_STOCKS_RUIN_FLOOR}, equal weight puts more "
            f"than 10% behind each name — above the SEBI single-stock cap this "
            f"IPS applies. Relax the score gate, drop a sector exclusion, or "
            f"increase the SIP.")

    # Infeasible => shrink n. NEVER lower the gate, never inject a stock that
    # failed Tier 2 to satisfy a percentage.
    q = _quotas(n, alloc)
    ok, why = _feasible(pool, q)
    while not ok and n > 1:
        n -= 1
        q = _quotas(n, alloc)
        ok, why = _feasible(pool, q)
    if n < min(len(pool), aff_n, ips_target):
        warnings.append(f"Portfolio shrunk to {n} holdings: {why}.")

    chosen, slot_type = _fill(pool, q, corr, k_conviction=k_conv)

    # Rounded for display, but the validator re-normalizes off this value, so a
    # 1-decimal round at n=12 (8.3) reintroduces the epsilon it was meant to
    # avoid: 12 x 8.3 = 99.6, not 100. Keep enough precision that the sum is
    # exact to well inside the tolerance. Renderers should format, not the model.
    pct = round(100.0 / max(len(chosen), 1), 4)
    holdings = []
    for idx in chosen:
        r = pool.loc[idx]
        app = r["_applicable"]
        abstained = [f for f in FRAMEWORKS if f not in app]
        holdings.append({
            "ticker": r["ticker"], "name": r.get("name", ""),
            "sector": r["sector"], "score": int(r.get("score", 0)),
            "price": float(r["price"]), "risk_tier": r["risk_tier"],
            "allocation_pct": pct,
            "pe": r.get("pe"), "roe_pct": r.get("roe_pct"), "beta": r.get("beta"),
            "_trace": {
                # Two ORTHOGONAL facts. Conflating them is why every stock used
                # to read "diversifier": the old role was a function of
                # diversification_rank and nothing else.
                # Two ORTHOGONAL facts, and both are now informative.
                "slot_type": slot_type[idx],           # why it got a seat
                "gate_cleared": ("conviction_sleeve" if slot_type[idx] == "conviction"
                                 else "abstention_adjusted" if abstained
                                 else "merit"),        # how it cleared Tier 2
                "conviction_framework": (dom_framework if slot_type[idx] == "conviction"
                                         else None),
                "conviction_pct": (round(float(r["_conviction_pct"]), 3)
                                   if slot_type[idx] == "conviction"
                                   and not pd.isna(r.get("_conviction_pct"))
                                   else None),
                # The rank that CHOSE this stock, not the composite rank that
                # buried it. "#3 of 56 on Greenblatt in Basic Materials."
                "conviction_rank": (int(r["_conviction_rank"])
                                    if slot_type[idx] == "conviction"
                                    and not pd.isna(r.get("_conviction_rank"))
                                    else None),
                "sector": r["sector"],
                "rank_in_sector": int(r["_rank_in_sector"]),
                "sector_depth": int(r["_sector_depth"]),
                "rank_score": round(float(r["_rank_score"]), 4),
                "applicable": list(app),
                "ranked_on": list(r["_ranked_on"]),
                "abstained": abstained,
                "score_applicable": int(r["_score_applicable"]),
                "effective_gate": int(r["_effective_gate"]),
                "passed": [f for f in app if bool(r.get(PASS_FLAG[f], False))],
                "failed": [f for f in app if not bool(r.get(PASS_FLAG[f], False))],
                "tiebreak_metric": r["_tiebreak_metric"],
                "tiebreak_value": (None if pd.isna(r["_tiebreak_value"])
                                   else round(float(r["_tiebreak_value"]), 3)),
                # ── W1 drift-decomposition inputs ──────────────────────────
                # Captured HERE because _trace is the only thing app.py
                # persists as entry_trace; the top-level pe/roe_pct on this
                # holding dict are dropped at insert, and holdings.pe_at_entry /
                # roe_at_entry are re-derived later from different columns
                # (roe_y0, a re-fetched row) so they are NOT a comparable
                # entry side. Recorded at SELECTION time means entry and
                # current are produced by the same code path.
                "pe": _num(r.get("pe"), 3),
                "roe_pct": _num(r.get("roe_pct"), 3),
                "score_continuous": _num(r.get("score_continuous")),
                "fracs": {f: _num(r.get(col)) for f, col in FRAC_COL.items()},
                # -- Term-level attribution inputs ------------------------
                # INPUTS, not terms. Both sides get re-derived with today's
                # framework_terms(), so a threshold change can never read as a
                # business change. terms_version guards the comparison: an
                # older vector is "not comparable", never a term scoring zero.
                "term_inputs": {c: _jsonable(r.get(c)) for c in TERM_INPUT_COLS},
                "terms_version": TERMS_VERSION,
            },
        })

    # Rejections build more trust than the picks do.
    # "HDFCBANK ranked #1 in Financials; sector already at 3/3."
    rejections = []
    for sec, grp in pool[~pool.index.isin(chosen)].groupby("sector"):
        for _, r in grp.nsmallest(2, "_rank_in_sector").iterrows():
            rejections.append({
                "ticker": r["ticker"], "sector": sec,
                "rank_in_sector": int(r["_rank_in_sector"]),
                "reason": "sector_full" if any(h["sector"] == sec for h in holdings)
                          else "outranked_globally",
            })

    return {
        "holdings": holdings,
        "warnings": warnings,
        "rejects": rejects,
        "rejections": rejections[:12],
        "diagnostics": {
            "pool_size": len(pool),
            "affordable_n": aff_n,
            "ips_target": ips_target,
            "n_selected": len(chosen),
            "quotas": q,
            "weights": _resolve_weights(policy),
            "philosophy": policy.get("philosophy"),
            "min_acceptable_score": policy.get("min_acceptable_score"),
            "acceptable_tradeoff": policy.get("acceptable_tradeoff"),
            "demand_tilt": policy.get("demand_tilt"),
            "covariance_used": corr is not None,
            "conviction_slots": k_conv,
            "conviction_framework": dom_framework,
            "conviction_candidates": int(pool["_conviction_eligible"].sum()),
            "sector_counts": {h["sector"]: sum(1 for x in holdings if x["sector"] == h["sector"])
                              for h in holdings},
            "tier_counts": {t: sum(1 for x in holdings if x["risk_tier"] == t)
                            for t in ("Large", "Mid", "Small", "Micro")},
        },
    }

def investable_tickers(universe_df: pd.DataFrame, sip_amount: float,
                       avoid_sectors=None, limit: int = 250) -> list[str]:
    """Tier-1 survivors, best-scored first. Callers use this to decide which
    price histories to download — WITHOUT reimplementing the floor.

    Single source of truth: if MIN_TURNOVER changes, this changes with it.
    """
    df = _tier1(universe_df.copy(), sip_amount, {})
    if avoid_sectors:
        df = df[~df["sector"].isin(set(avoid_sectors))]
    return (df.sort_values("score", ascending=False)
              .head(limit)["ticker"].tolist())

# ══════════════════════════════════════════════════════════════════════════
# LANDING SCREEN — improving businesses, not perfect scores
# ══════════════════════════════════════════════════════════════════════════
# NOT the 5-of-5 list. A watchlist exists to be WATCHED, and the alert that
# brings a user back is watchlist_score_up. A perfect score has nowhere to go
# but down. A 3-of-5 that fails only Graham becomes a 4-of-5 the day its price
# falls — which teaches the correct reflex: a price drop is GOOD news, because
# it widens margin of safety.
#
# We do NOT claim "the market hasn't noticed yet". We measured that claim and
# could not support it. `pe_vs_avg` is a percent deviation from a stock's own
# 4-year average P/E, and P/E has EARNINGS in the denominator — so any stock we
# selected on trajectory and sorted by earnings CAGR shows a depressed P/E
# versus its own history *by construction*. Bharti Airtel reads -55 while
# sitting at an all-time high. The metric measured the growth we conditioned on.
#
# The honest claim is the failure pattern: graham appeared in 65 of the 73
# three-of-five stocks, exactly as its 5.6% base rate predicts.
#     "These businesses are measurably improving. Graham would tell you they
#      aren't cheap. Both are true, and which matters is up to you."
MIN_BASE_NET_INCOME = 10e7   # ₹10 crore


def improving_businesses(universe_df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    """Investable, quality-passing, improving, and NOT unanimous."""
    df = _tier1(universe_df.copy(), sip_amount=float("inf"), rejects={})
    df = _attach_applicable(df)

    df["attainable"] = df["_applicable"].apply(len)
    df["score_applicable"] = [
        sum(1 for f in r["_applicable"] if bool(r.get(PASS_FLAG[f], False)))
        for _, r in df.iterrows()
    ]
    df["abstained"] = df["_applicable"].apply(
        lambda a: ", ".join(f for f in FRAMEWORKS if f not in a))
    df["failed"] = [
        ", ".join(f for f in r["_applicable"] if not bool(r.get(PASS_FLAG[f], False)))
        for _, r in df.iterrows()
    ]

    m = (df["trajectory_pass"].fillna(False).astype(bool)
         # 3 .. attainable-1: strong, and not unanimous. Excludes perfect scores.
         & (df["score_applicable"] >= 3)
         & (df["score_applicable"] <= df["attainable"] - 1)
         # Base-effect guard. Nobody compounds earnings at 226% for three years:
         # SAILIFE, WABAG and PRIVISCL are recoveries off a near-zero base, and
         # sorting on ni_cagr_3y would put the noisiest names on the front page.
         & (df["net_income_y3"].fillna(0) >= MIN_BASE_NET_INCOME))

    # revenue_cagr_3y, never ni_cagr_3y. Revenue has no zero-base pathology.
    return (df[m]
            .sort_values(["trajectory_score", "revenue_cagr_3y", "pe"],
                         ascending=[False, False, True])
            .head(limit))

# ══════════════════════════════════════════════════════════════════════════
# THESIS DRIFT  (Sprint 13 §2)
#
# A review must show what CHANGED since purchase, not re-explain the holding.
# diff_thesis compares the recorded reason a stock was bought (its entry _trace,
# stored on holdings at registration) against its status in a fresh
# select_portfolio run over today's universe. Pure: no Streamlit, no network,
# no LLM. The caller renders the diff; the LLM, if used at all, only phrases
# these deterministic facts into prose — it never decides the classification.
# ══════════════════════════════════════════════════════════════════════════

# The one slot_type that means "held only because the conviction sleeve fired" —
# a specialist the merit passes left behind. The other four (cap_quota_large,
# cap_quota_mid, breadth, free) all earned a seat without the sleeve.
_CONVICTION_SLOTS = {"conviction"}

# W1. How each framework is ALLOWED to be read when it flips. Set a priori from
# what the framework's own arithmetic contains, never from observed data.
#   fundamental — no meaningful price input; a flip is the business moving.
#   mixed       — contains BOTH price and fundamentals; must be disambiguated
#                 from pe/roe before any reason is named.
# dorsey_buffett is 'fundamental' with one honest caveat: nine of its ten checks
# are pure fundamentals, and the tenth (buffett_one_dollar_test) is a 4-YEAR
# trailing market-cap delta. It is a price term. It is slow and boolean, so it
# rarely drives a flip, but the label is a dominant reading, not a purity claim.
# greenblatt is 'mixed', NOT 'relative-rank': greenblatt_frac is the percentile
# of (roic_rank + ey_rank) and ey = EBIT/EV, so it moves on this stock's price,
# this stock's EBIT/ROIC, or other stocks. Calling it relative-rank up front
# would tell a user "nothing about your stock changed" while the price halved.
# RETIRED 2026-07-31. This table declared each framework's cause in advance,
# and _classify_flip returned on it BEFORE looking at any input - so a Dorsey
# flip driven entirely by buffett_one_dollar_test (a 4-year market-cap delta,
# which moved on 87.4% of no-news rows) was labelled "fundamental" with no
# price test ever run. Cause is now measured per row from the term that moved
# and the components under it. Nothing declares a framework's cause any more.
#
# Measured on the live book before removal: of 8 down-flips, 7 were labelled
# `unclear` -> danger because roe_pct (annual, from info) had not moved on 26
# of 29 holdings and pe had cleared 10% on exactly 1. The residual case was
# the modal case.

# Where a 'mixed' framework flips but NEITHER pe nor roe moved. For greenblatt
# that is the informative case — its own inputs held, so the universe moved
# around it. For graham/lynch it means something we do not capture moved, and
# the only honest label is that we cannot tell.
# RETIRED with it. greenblatt's relative_rank survives as a MEASURED outcome
# rather than a default: see _greenblatt_flip.

# The noise floor for "this input MOVED". NOT a causal test — proving pe caused
# a flip needs metric-level archive rows (Tier 2, gated to ~2027). This only
# separates a move worth naming from rounding and refresh jitter.
# JUDGMENT, not a fitted parameter: 10% sits below a typical quarterly earnings
# revision and above CSV rounding. Change it by argument. Never fit it.
DRIFT_MATERIAL_PCT = 0.10

# ── Two drift floors, two denominators ────────────────────────────────────
# One constant served both sites. They are not on the same scale:
#     per-framework frac delta   lives on [-1, 1]
#     total score_continuous     lives on [-5, 5]
# so 0.05 asserted "5% of range" at one site and "1% of range" at the other.
# Same error class as the abstention denominators and the raw-score gates: a
# threshold counted against one denominator applied to another.
#
# OPEN (backlog): this floor is not uniform in EFFECT. trajectory_graded is
# all-boolean in steps of 1-2, so its smallest possible frac move is 0.1 and
# the floor never binds on it; graham/greenblatt/dorsey/lynch carry ramps and
# move continuously. greenblatt_frac is a universe percentile recomputed every
# run with ey = EBIT/EV on price, so it moves daily with no fundamental change.
# A single floor therefore biases largest_move toward the coarsest framework.
# Fixing it needs the per-framework delta distribution across consecutive
# archive runs — a measurement, not an argument. Not fitted here.
DRIFT_FLOOR_FRAMEWORK = 0.05

# DERIVED, not chosen. If every framework moves just under the floor, the total
# is len(FRAMEWORKS) x (floor - eps). Below that bound an alert could be
# composed entirely of moves the per-framework site calls immaterial, and
# continuous_drift would emit with largest_move = None — "your score moved,
# cause unattributable." Sign-robust: all |d| < floor gives |sum d| < bound
# whatever the signs. Expressed rather than literal so that per-framework
# floors, when measured, carry the total with them instead of drifting apart.
DRIFT_MATERIAL_TOTAL = len(FRAMEWORKS) * DRIFT_FLOOR_FRAMEWORK

# Worst-case wins. Index 0 is worst. When several frameworks flip at once and
# disagree about why, a single fundamental break makes the whole drop
# fundamental: a stock can get expensive AND deteriorate in the same quarter,
# and the deterioration is the part that costs money.
DRIFT_PRECEDENCE = (
    "fundamental",
    "mixed",
    "departed",
    "unclear",
    "unknown_inputs",
    "valuation",
    "relative_rank",
)

# Only the two labels that POSITIVELY establish "the business held" may soften
# an alert. Both flavours of "we could not tell" keep the existing severity.
# Absence of evidence must never downgrade — otherwise every holding bought
# before W1 shipped, which is all of them, gets quietly demoted on a label that
# means nothing more than "no data".
DRIFT_SEVERITY = {
    "fundamental":    "danger",
    "mixed":          "danger",
    # A framework that left the comparison set. NOT a failure - but not benign
    # either: one way to leave is to post a loss year, lose the archetype and
    # drop out of Lynch, which is deterioration. Absence of evidence must not
    # downgrade, and a departure is evidence that something moved.
    "departed":       "danger",
    "unclear":        "danger",
    "unknown_inputs": "danger",
    "valuation":      "warning",
    "relative_rank":  "warning",
}

# Watchlist ceiling is WARNING. Severity measures capital at risk, and a watched
# stock is not owned — nothing is at stake but an intention. Same shape as
# DRIFT_SEVERITY otherwise: only the two labels that positively establish "the
# business held" soften anything, and both "we could not tell" labels keep the
# existing level.
WATCHLIST_DRIFT_SEVERITY = {
    "fundamental":    "warning",
    "mixed":          "warning",
    "unclear":        "warning",
    "unknown_inputs": "warning",
    "valuation":      "info",
    "relative_rank":  "info",
}

# ── W2: archetype conditions SEVERITY, never the label ────────────────────
# _classify_flip answers WHICH INPUT MOVED and must keep answering only that —
# its docstring is explicit that favourability lives in newly_passing vs
# newly_failing, and duplicating it there would let the two disagree. So the
# archetype enters one step later, at "how alarmed should you be".
#
# ONE CELL, and it is Lynch's own inversion. "valuation" normally means price
# moved and the business held, hence only a warning. For a CYCLICAL that reading
# flips: a falling PE with fundamentals apparently intact is the peak-earnings
# signature — earnings follow the price down, and the low PE is the trap, not
# the discount. lynch_score's cyclical branch already encodes the same claim.
#
# HARDEN-ONLY, and this is what keeps the table from growing into 6x6 = 36
# cells of pairwise overfit. An archetype is CONTEXT, not evidence that a
# business held, so it may raise an alarm and must never quiet one. Same
# principle as the note above DRIFT_SEVERITY: absence of evidence never
# downgrades.
#
# The other rows of the sourced sign-flip table (rising leverage on a cyclical,
# margin compression on a fast grower, inventory build, dividend cut on a
# stalwart) are NOT here because entry_trace carries only pe, roe_pct,
# score_continuous and the five fracs. There is no leverage, margin, inventory
# or dividend on either side of the comparison, so those rules cannot fire.
# Extending entry_trace is a jsonb schema change plus a backfill — deliberately
# out of W2 scope, recorded so the omission is not mistaken for an oversight.
_SEVERITY_ORDER = ("info", "warning", "danger")

ARCHETYPE_SEVERITY_FLOOR = {
    ("cyclical", "valuation"): "danger",
}


def apply_archetype_severity(severity, archetype, reason, ceiling=None):
    """Raise `severity` to the archetype floor for this reason, never lower it.

    ceiling: optional cap ("warning" on watchlist surfaces, where severity
    measures capital at risk and a watched stock is not owned). Applied AFTER
    the floor, so the watchlist ceiling still wins — a floor of danger on an
    unowned stock lands at warning, matching WATCHLIST_DRIFT_SEVERITY's own note.
    """
    if not severity or not archetype or not reason:
        return severity
    floor = ARCHETYPE_SEVERITY_FLOOR.get((archetype, reason))
    if floor is None:
        return severity
    try:
        out = max(severity, floor, key=_SEVERITY_ORDER.index)
    except ValueError:
        return severity            # unknown label: leave it alone, never soften
    if ceiling:
        try:
            if _SEVERITY_ORDER.index(out) > _SEVERITY_ORDER.index(ceiling):
                out = ceiling
        except ValueError:
            pass
    return out


def _on_conviction(trace: dict) -> bool:
    return bool(trace) and trace.get("slot_type") in _CONVICTION_SLOTS


def _trace_facts(trace: dict) -> dict:
    """The subset of a _trace used to render a thesis line. Defensive to
    partial/older traces and to manual holdings with no trace at all."""
    if not trace:
        return {}
    return {
        "slot_type": trace.get("slot_type"),
        "gate_cleared": trace.get("gate_cleared"),
        "conviction_framework": trace.get("conviction_framework"),
        "conviction_rank": trace.get("conviction_rank"),
        "conviction_pct": trace.get("conviction_pct"),
        "sector": trace.get("sector"),
        "rank_in_sector": trace.get("rank_in_sector"),
        "sector_depth": trace.get("sector_depth"),
        "score_applicable": trace.get("score_applicable"),
        "effective_gate": trace.get("effective_gate"),
        "passed": list(trace.get("passed", [])),
        "failed": list(trace.get("failed", [])),
        # W1 decomposition inputs. Absent on any trace written before edit 1 —
        # .get() -> None, which every consumer below treats as "unmeasured",
        # never as "held". No backfill; this self-heals as holdings turn over.
        "pe": trace.get("pe"),
        "roe_pct": trace.get("roe_pct"),
        "score_continuous": trace.get("score_continuous"),
        "fracs": dict(trace.get("fracs") or {}),
        # Absent on any trace written before step 3 -> None, which _term_diff
        # reports as NOT comparable. Never as "the inputs held".
        "term_inputs": trace.get("term_inputs"),
        "terms_version": trace.get("terms_version"),
    }


def _moved(a, b, floor: float = DRIFT_MATERIAL_PCT):
    """Did a tracked input move beyond the noise floor?

    True / False / None, and the None matters: a missing side means we do not
    know, which is emphatically not the same as "it held". Collapsing the two
    is how a data gap gets reported to a user as a stable business."""
    if a is None or b is None:
        return None
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return None
    if a == 0:
        return b != 0
    return abs(b - a) / abs(a) >= floor


def _term_diff(framework: str, e: dict, c: dict):
    """Which terms changed integer POINTS between entry and today.
 
    Integer, not graded: compute_framework_verdicts thresholds the integer
    sub-score into the pass flag, so the integer half is what fires an alert.
    Where a term's two halves read different columns (D2 alone), attribution
    follows the integer input.
 
    Both sides are re-derived with TODAY's framework_terms() from stored
    INPUTS. Storing terms instead would mean comparing points computed under
    two different rule sets, so a threshold change would read as a business
    change — the break MIN_RECONCILABLE_SCHEMA exists to prevent.
 
    Returns (changed, comparable). comparable is False when either side cannot
    be re-derived; the caller must then say so, never assume the inputs held.
    """
    e_in, c_in = e.get("term_inputs"), c.get("term_inputs")
    if not isinstance(e_in, dict) or not isinstance(c_in, dict):
        return [], False
    if e.get("terms_version") != c.get("terms_version"):
        return [], False
    et = framework_terms(e_in).get(framework)
    ct = framework_terms(c_in).get(framework)
    if not et or not ct or set(et) != set(ct):
        # Different term sets = a Lynch category change. The yardstick moved,
        # not the business; that is a reclassification, reported by the caller.
        return [], False
    changed = []
    for name in sorted(et):
        if et[name][0] == ct[name][0]:
            continue
        moves = [{"column": col, "from": e_in.get(col), "to": c_in.get(col)}
                 for col in et[name][2]
                 if not _same_value(e_in.get(col), c_in.get(col))]
        changed.append({"term": name, "from": et[name][0], "to": ct[name][0],
                        "inputs": moves})
    return changed, True
 
 
def _same_value(a, b):
    """Equality that does not mistake 2.0 for '2.0' or NaN for a difference."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b
 
 
def _verdict(changed, e_in, c_in):
    """price vs fundamental, MEASURED on this row from component movement.
 
    A ratio input (pe, PEG, dividend yield, NCAV ratio) moves when its price
    component moves or when its fundamental component moves, and those mean
    different things to an owner. TERM_INPUT_COMPONENTS names the components;
    `price` is the price one and there is no other list.
 
    Returns None where the evidence does not separate — an unattributable
    input, or nothing measurable moved. None means "no verdict", and the
    caller reports the DELTA alone. A guessed category would be worse than
    the number.
    """
    price_moved = fund_moved = False
    for ch in changed:
        for m in ch["inputs"]:
            col = m["column"]
            if col in UNATTRIBUTABLE_INPUTS:
                return None
            comps = TERM_INPUT_COMPONENTS.get(col)
            if comps is None:
                # Not a ratio: the input is the thing it measures.
                if col == "price":
                    price_moved = True
                else:
                    fund_moved = True
                continue
            for comp in comps:
                if _moved(e_in.get(comp), c_in.get(comp)):
                    if comp == "price":
                        price_moved = True
                    else:
                        fund_moved = True
    if price_moved and fund_moved:
        return "mixed"
    if price_moved:
        return "valuation"
    if fund_moved:
        return "fundamental"
    return None
 
 
def _greenblatt_flip(e: dict, c: dict):
    """Greenblatt has no terms — one universe percentile — so it is attributed
    from its own two stored components, by arithmetic rather than declaration.
 
    roic = EBIT / tangible capital;  ey = EBIT / EV.  EBIT is common to both.
    So ey moving while roic holds means EV moved, i.e. price; roic moving means
    tangible capital moved; both moving means EBIT moved and cannot be split.
    Neither moving while the rank moved is the informative case: this stock's
    inputs held and the universe moved around it.
    """
    e_in, c_in = e.get("term_inputs"), c.get("term_inputs")
    if not isinstance(e_in, dict) or not isinstance(c_in, dict):
        return "unknown_inputs", {"terms": [], "comparable": False}
    ey = _moved(e_in.get("greenblatt_earnings_yield"),
                c_in.get("greenblatt_earnings_yield"))
    roic = _moved(e_in.get("greenblatt_roic"), c_in.get("greenblatt_roic"))
    detail = {"terms": [], "comparable": True,
              "components": {"greenblatt_earnings_yield":
                             [e_in.get("greenblatt_earnings_yield"),
                              c_in.get("greenblatt_earnings_yield")],
                             "greenblatt_roic": [e_in.get("greenblatt_roic"),
                                                 c_in.get("greenblatt_roic")]}}
    if ey is None or roic is None:
        return "unknown_inputs", detail
    if ey and roic:
        return "mixed", detail
    if ey:
        return "valuation", detail
    if roic:
        return "fundamental", detail
    return "relative_rank", detail
 
 
def _flip_detail(framework: str, e: dict, c: dict):
    """Why did this framework flip? (label, detail).
 
    detail carries the actual deltas — "Lynch lost 3 points: PEG 0.91 -> 1.62"
    — which is the primary output. The label is a coarse summary that ABSTAINS
    when the evidence does not separate; the delta never abstains.
 
    Vocabulary (closed set):
      fundamental    a fundamental component moved
      valuation      price moved and no fundamental component did
      mixed          both moved
      relative_rank  this stock's inputs held; the universe moved around it
      unclear        terms changed, nothing measurable moved under them
      unknown_inputs cannot re-derive one side; we cannot split it
    """
    if framework == "greenblatt":
        return _greenblatt_flip(e, c)
    changed, comparable = _term_diff(framework, e, c)
    if not comparable:
        return "unknown_inputs", {"terms": [], "comparable": False}
    detail = {"terms": changed, "comparable": True}
    v = _verdict(changed, e.get("term_inputs") or {}, c.get("term_inputs") or {})
    return (v or "unclear"), detail
 
 
def _departure_cause(f: str, e: dict, c: dict):
    """Why did `f` leave the comparison set? A departure is NOT a failure, but
    it is not benign: losing the archetype (a loss year kills ni_cagr_3y) is
    deterioration, and going dark on sector is a data loss. Both are reported;
    neither is counted as a framework the business failed."""
    e_in, c_in = e.get("term_inputs") or {}, c.get("term_inputs") or {}
    out = {"framework": f, "was_passing": f in set(e.get("passed") or [])}
    for col in ("lynch_category", "sector"):
        a = e.get(col) if col == "sector" else e_in.get(col)
        b = c.get(col) if col == "sector" else c_in.get(col)
        if not _same_value(a, b):
            out[col] = [a, b]
    out["kind"] = "inputs_missing" if any(
        v[1] is None for k, v in out.items()
        if isinstance(v, list) and len(v) == 2) else "inputs_moved"
    return out


def _continuous_drift(e: dict, c: dict) -> dict | None:
    """Decompose delta(score_continuous) into per-framework contributions.

    score_continuous is the SUM of the five fracs, so the split is exact —
    when every frac is measured on both sides. When one is not, that framework
    is excluded rather than guessed, and the shortfall is reported as
    `unattributed` instead of being silently absorbed into the others.

    This gap is real and must stay visible. deep_metrics coerces a None frac to
    0 inside score_continuous, so a stock that merely BECOMES rankable (a filing
    lands, a throttled fetch succeeds) shows a ~1-point jump with nothing about
    the business having changed — and the reverse reads as a collapse. Naming it
    `unattributed` is what stops that arithmetic becoming a thesis signal."""
    ef, cf = (e.get("fracs") or {}), (c.get("fracs") or {})
    e_sc, c_sc = e.get("score_continuous"), c.get("score_continuous")
    if e_sc is None or c_sc is None:
        return None
    try:
        e_sc, c_sc = float(e_sc), float(c_sc)
    except (TypeError, ValueError):
        return None
    total = round(c_sc - e_sc, 4)

    by_framework, unmeasured = {}, []
    for f in FRAMEWORKS:
        a, b = ef.get(f), cf.get(f)
        if a is None or b is None:
            unmeasured.append(f)
            continue
        try:
            by_framework[f] = round(float(b) - float(a), 4)
        except (TypeError, ValueError):
            unmeasured.append(f)

    attributed = round(sum(by_framework.values()), 4)
    largest = (max(by_framework, key=lambda k: abs(by_framework[k]))
               if by_framework else None)
    # A "largest move" smaller than the noise floor is not a mover. Report none
    # rather than crown the biggest rounding error in the set.
    if largest is not None and abs(by_framework[largest]) < DRIFT_FLOOR_FRAMEWORK:
        largest = None

    return {
        "from": round(e_sc, 4),
        "to": round(c_sc, 4),
        "delta": total,
        "by_framework": by_framework,
        "largest_move": largest,
        "unmeasured": unmeasured,
        "unattributed": round(total - attributed, 4),
    }


def _thesis_changes(entry: dict, current: dict) -> list:
    """Ordered, deterministic list of what moved between entry and today.
    Each item is {"field", "from", "to"}. Empty when nothing tracked changed."""
    changes = []
    e, c = _trace_facts(entry), _trace_facts(current)

    # Rank within the sector pool — the margin narrowing or widening.
    if (e.get("rank_in_sector"), e.get("sector_depth")) != \
       (c.get("rank_in_sector"), c.get("sector_depth")):
        changes.append({"field": "rank_in_sector",
                        "from": (e.get("rank_in_sector"), e.get("sector_depth")),
                        "to": (c.get("rank_in_sector"), c.get("sector_depth"))})

    # How many applicable frameworks it passes now.
    if e.get("score_applicable") != c.get("score_applicable"):
        changes.append({"field": "score_applicable",
                        "from": e.get("score_applicable"),
                        "to": c.get("score_applicable")})

    # Which specific frameworks flipped, in each direction. `to` stays a plain
    # list of names — existing renderers join it directly. The reason labels ride
    # alongside in a NEW key, so nothing that reads the old shape breaks.
    e_pass, c_pass = set(e.get("passed", [])), set(c.get("passed", []))
    _flips = {f: _flip_detail(f, e, c) for f in (e_pass | c_pass)}
    _gained = sorted(c_pass - e_pass)
    if _gained:
        changes.append({"field": "newly_passing", "from": None, "to": _gained,
                        "reasons": {f: _flips[f][0] for f in _gained},
                        "detail": {f: _flips[f][1] for f in _gained}})
    _lost = sorted(e_pass - c_pass)
    if _lost:
        changes.append({"field": "newly_failing", "from": None, "to": _lost,
                        "reasons": {f: _flips[f][0] for f in _lost},
                        "detail": {f: _flips[f][1] for f in _lost}})

    # Conviction rank drift, when both entry and today were conviction picks.
    if e.get("conviction_rank") is not None and c.get("conviction_rank") is not None \
       and e.get("conviction_rank") != c.get("conviction_rank"):
        changes.append({"field": "conviction_rank",
                        "from": e.get("conviction_rank"),
                        "to": c.get("conviction_rank")})

    # The seat itself changed character (e.g. large-cap floor -> pure merit).
    if e.get("slot_type") != c.get("slot_type"):
        changes.append({"field": "slot_type",
                        "from": e.get("slot_type"), "to": c.get("slot_type")})

    # Magnitude, not just direction. Gated on the TOTAL bound, not the
    # per-framework floor — those are different denominators and comparing the
    # total against the framework floor is what let a 0.06 move spread across
    # five sub-floor wobbles render as news with no nameable cause. Returns
    # None on pre-edit-1 traces, which is the correct silence.
    cd = _continuous_drift(e, c)
    if cd and abs(cd["delta"]) >= DRIFT_MATERIAL_TOTAL:
        changes.append({"field": "continuous_drift",
                        "from": cd["from"], "to": cd["to"], "detail": cd})

    return changes


def diff_thesis(entry_trace: dict | None, current_trace: dict | None,
                still_investable: bool = True) -> dict:
    """
    Classify how a holding's thesis has drifted since purchase.

    entry_trace     : _trace stored at registration, or None if never recorded
                      (manual holding, or bought before trace capture shipped).
    current_trace   : the same ticker's _trace in a fresh selection over today's
                      universe, or None if it was not selected today.
    still_investable: is the ticker still in the Tier-1 pool today? Only consulted
                      when current_trace is None, to tell "outranked" (still
                      investable, just not top-N) apart from "fell out of the
                      pool". Defaults True so a caller that cannot check pool
                      membership never falsely claims the turnover floor failed.

    Returns {"drift", "entry", "current", "changes"} where drift is one of:
      no_trace              no entry thesis to compare against.
      no_longer_investable  fell out of the Tier-1 pool (turnover/quality floor);
                            would NOT be bought today.
      outranked             still investable, but other names now rank above it;
                            not in today's portfolio.
      now_merit             was held on conviction, now clears the gate on merit
                            (thesis strengthened).
      now_conviction        was a merit pick, now survives only via the
                            conviction sleeve (thesis weakened).
      still_selected        same basis; see `changes` for the delta.
    """
    if not entry_trace:
        return {"drift": "no_trace", "entry": {},
                "current": _trace_facts(current_trace), "changes": []}

    if not current_trace:
        # Not selected in today's re-run. Distinguish two very different
        # realities: still in the pool but outranked (soft) vs fell out of the
        # pool entirely — the turnover/quality floor, the real sell signal.
        drift = "outranked" if still_investable else "no_longer_investable"
        return {"drift": drift,
                "entry": _trace_facts(entry_trace), "current": None, "changes": []}

    was_conv, now_conv = _on_conviction(entry_trace), _on_conviction(current_trace)
    drift = ("now_merit" if (was_conv and not now_conv)
             else "now_conviction" if (not was_conv and now_conv)
             else "still_selected")

    return {"drift": drift,
            "entry": _trace_facts(entry_trace),
            "current": _trace_facts(current_trace),
            "changes": _thesis_changes(entry_trace, current_trace)}


def compute_thesis_drift(holdings, policy, universe_df, price_history=None):
    """
    Diff every held position's stored entry thesis against a fresh selection
    over today's universe. Pure: re-runs select_portfolio + investable_tickers,
    no Streamlit and no network (unless a price_history is passed in). Returns
    {ticker: diff_thesis(...)}.

    holdings : stored holding dicts, each with "ticker" and ideally
               "entry_trace" (missing -> drift "no_trace").
    policy   : the IPS policy the portfolio was built under
               (portfolio_profile.ips_policy). Falsy -> {}: drift is undefined
               without the mandate that decides "would we buy this today".

    price_history is left None on the review path by design. It only feeds the
    correlation tiebreak, which nudges _rank_score at the margin; it does not
    change pool membership or the conviction/merit split, which is what the
    drift classes turn on. Passing None keeps the review fast and deterministic.
    """
    if not policy or universe_df is None or not len(universe_df):
        return {}
    try:
        result = select_portfolio(universe_df, policy, price_history)
    except Exception:
        return {}
    current_by_ticker = {h["ticker"]: h.get("_trace") for h in result.get("holdings", [])}

    sip = policy.get("sip_amount", 0) or 0
    avoid = policy.get("avoid_sectors", []) or []
    try:
        # limit=len(universe_df) => the FULL Tier-1 pool, not the top-250 slice.
        # We need true pool membership to tell "outranked" from "fell out".
        pool = set(investable_tickers(universe_df, sip, avoid, limit=len(universe_df)))
    except Exception:
        pool = set()

    out = {}
    for h in holdings:
        tkr = h.get("ticker")
        if not tkr:
            continue
        out[tkr] = diff_thesis(h.get("entry_trace"),
                               current_by_ticker.get(tkr),
                               still_investable=(tkr in pool))
    return out

def _current_facts(universe_row):
    """Today's comparison facts, plus which frameworks pass, from a universe row.

    Shaped exactly like a _trace so entry and current sides go through identical
    comparison logic in _classify_flip.
    """
    row = universe_row if universe_row is not None else {}

    def _get(col):
        try:
            v = row.get(col)
        except AttributeError:
            return None
        try:
            if v is None or pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v

    current = {
        "pe": _num(_get("pe"), 3),
        "roe_pct": _num(_get("roe_pct"), 3),
        "score_continuous": _num(_get("score_continuous")),
        "fracs": {f: _num(_get(col)) for f, col in FRAC_COL.items()},
        # Every column the term tables read, plus the components of each ratio
        # among them. Unrounded: these are re-scored, not displayed. _jsonable,
        # NOT _get: build_watch_trace's output is persisted straight to Supabase
        # with no sanitiser (app.py:2997, 6939), and a numpy.bool_ or float64
        # off a typed DataFrame raises on json.dumps. It also keeps this side
        # byte-symmetric with the entry side, which is written with _jsonable.
        "term_inputs": {col: _jsonable(_get(col)) for col in TERM_INPUT_COLS},
        "terms_version": TERMS_VERSION,
        "sector": _get("sector"),
    }
    applicable = set(_applicable_frameworks(row))
    passing = {f for f in FRAMEWORKS
               if f in applicable and bool(_get(PASS_FLAG[f]))}
    return current, applicable, passing


def _label_flips(frameworks, entry, current, applicable):
    """Per-framework labels AND deltas for one direction, worst-case-wins.
    `frameworks` is already restricted to the COMPARABLE set by the caller, so
    the old `f not in applicable -> unclear` branch is gone. It was written when
    sector exclusion was the only way to leave `applicable` and called the case
    "near-impossible"; v6 made Lynch abstention routine (2,328 rows) and it
    became 2 of 8 live down-flips, each labelled `unclear` -> danger. A
    departure is now its own event with its own cause, never a flip.
    """
    per, detail = {}, {}
    for f in sorted(frameworks):
        per[f], detail[f] = _flip_detail(f, entry, current)
    if not per:
        return {}, {}, None
    # Unrecognised labels sort as WORST, never mildest: if _classify_flip grows
    # a label nobody mapped, the failure direction must be louder, not quieter.
    reason = min(per.values(),
                 key=lambda r: DRIFT_PRECEDENCE.index(r)
                 if r in DRIFT_PRECEDENCE else -1)
    return per, detail, reason


def classify_score_change(entry_trace: dict | None, universe_row) -> dict:
    """Which frameworks flipped since entry, in BOTH directions, and why.

    Shared by holdings (score_drop) and watchlist entries (score_up/down).
    Path B: no selection re-run — entry_trace plus today's universe row is
    everything needed.

    Returns {"traceable": bool, "down": {...}, "up": {...}} where each
    direction carries {frameworks, per_framework, reason}; "down" also carries
    "severity".

    reason is None when a direction has no flips. That is NOT "we could not
    tell" — it is "nothing happened here". Only a caller that KNOWS a change
    occurred can translate None into 'unclear', so that translation lives in
    the caller, not here.

    Severity is down-side only. There is no severity above 'info' for good
    news, so on the up side the label shapes the MESSAGE and decides nothing.
    """
    current, applicable, passing = _current_facts(universe_row)
    entry = entry_trace or {}
    entry_passed = entry.get("passed")

    if not isinstance(entry_passed, (list, tuple)) or not entry_passed:
        # No recorded entry thesis. We cannot name a cause, and failing to name
        # one must not be mistaken for naming a benign one.
        blank = {"frameworks": [], "per_framework": {},
                 "reason": "unknown_inputs"}
        return {"traceable": False,
                "down": {**blank, "severity": DRIFT_SEVERITY["unknown_inputs"]},
                "up": dict(blank)}

    # Compare on the frameworks applicable to BOTH sides - the same set the
    # firing gate uses (comparable_score_drop). The gate and the explanation
    # disagreeing is what put LUPIN and GREENPOWER on "lynch failed, danger"
    # when Lynch had abstained on them.
    e_app = entry.get("applicable")
    e_app = set(e_app) & set(FRAMEWORKS) if isinstance(e_app, (list, tuple)) \
        else set(FRAMEWORKS)
    common = (e_app & applicable) & set(FRAMEWORKS)
 
    ep = set(entry_passed) & set(FRAMEWORKS)
    newly_failing = (ep & common) - passing
    newly_passing = (passing & common) - ep
 
    # Frameworks that LEFT. Not failures - but not silent either: one way to
    # leave is a loss year killing ni_cagr_3y, losing the archetype, and
    # dropping out of Lynch, which is deterioration the user must hear about.
    departed = sorted(e_app - applicable)
    departures = [_departure_cause(f, entry, current) for f in departed]
 
    down_pf, down_dt, down_reason = _label_flips(newly_failing, entry, current,
                                                 applicable)
    up_pf, up_dt, up_reason = _label_flips(newly_passing, entry, current,
                                           applicable)
    if down_reason is None and departures:
        down_reason = "departed"
 
    return {
        "traceable": True,
        "terms_version": current.get("terms_version"),
        "comparable_frameworks": sorted(common),
        "departed": departures,
        "down": {"frameworks": sorted(newly_failing), "per_framework": down_pf,
                 "detail": down_dt, "reason": down_reason,
                 "severity": DRIFT_SEVERITY.get(down_reason) if down_reason else None},
        "up": {"frameworks": sorted(newly_passing), "per_framework": up_pf,
               "detail": up_dt, "reason": up_reason},
    }


def classify_score_drop(entry_trace: dict | None, universe_row) -> dict:
    """The DOWN half, for a caller that has already observed a real drop.

    A thin wrapper over classify_score_change. It had its own inline copy of
    the down-side logic until now — two implementations of the same rules,
    which is how a fix lands in one and not the other three months later.

    Contract unchanged: {reason, severity, newly_failing, per_framework}, with
    reason never None and severity never absent.
    """
    d = classify_score_change(entry_trace, universe_row)["down"]
    reason = d["reason"]
    if reason is None:
        # The caller saw the integer score fall; the framework diff does not
        # corroborate. score_at_entry and entry_trace['passed'] are stored
        # separately and can disagree. Say we cannot tell, rather than invent a
        # cause or quietly soften an alert we cannot explain.
        reason = "unclear"
    return {"reason": reason,
            "severity": DRIFT_SEVERITY.get(reason, "danger"),
            "newly_failing": d["frameworks"],
            "per_framework": d["per_framework"]}


def build_watch_trace(universe_row) -> dict:
    """Entry-side facts for a WATCHED stock, shaped like a holding's entry_trace.

    No slot_type and no sector rank: the stock was never selected, so the fields
    describing a selection seat do not apply. What it does carry is exactly what
    classify_score_change compares — the framework pass-set plus pe/roe/fracs/
    score_continuous.

    Lives here rather than in app.py because it encodes which COLUMN backs which
    framework (PASS_FLAG, FRAC_COL). app.py has two watchlist insert sites; each
    would otherwise carry its own copy of that mapping and they would drift.

    All values pass through _num, so the result is JSON-safe by construction —
    no NaN can reach the jsonb column, where it would compare unequal to itself
    and report a change that never happened.
    """
    current, applicable, passing = _current_facts(universe_row)
    return {"passed": sorted(passing),
            "failed": sorted(applicable - passing),
            # `applicable` was missing here, so comparable_score_drop could not
            # function on a watchlist trace at all - it fell to the raw-score
            # branch every time. Pre-existing; fixed with the rest.
            "applicable": sorted(applicable),
            **current}

# ══════════════════════════════════════════════════════════════════════════
# BENCHMARK SELECTION  (Sprint 13 §1)
#
# One benchmark per portfolio, matched to the cap profile the IPS MANDATES —
# not to what the selector picked. Written once at registration and never
# recomputed: re-deriving at review time is benchmark shopping (a portfolio
# that drifts small-cap would get re-benchmarked to a small-cap index and
# suddenly look good), and sip_transactions rows hold units of a SPECIFIC ETF,
# so switching the ticker later values one ETF's units at another's price.
#
# ETFs, not indices: the counterfactual is "what if I'd SIP'd into an index
# fund", so the benchmark must be a thing that trades — tracking error and
# expense ratio included. Tickers measured clean (1y history, 0 NaN, current)
# by probe_benchmark.py on 2026-07-13.
# ══════════════════════════════════════════════════════════════════════════
BENCHMARKS = {
    "nifty50":     {"ticker": "NIFTYBEES.NS",  "label": "Nifty 50"},
    "midcap150":   {"ticker": "MID150BEES.NS", "label": "Nifty Midcap 150"},
    "smallcap250": {"ticker": "SMALLCAP.NS",   "label": "Nifty Smallcap 250"},
}


def choose_benchmark(ips_policy: dict) -> dict:
    """
    Pick the ONE benchmark ETF whose cap profile matches the IPS mandate.
    Pure, deterministic. Returns {"ticker", "label", "reason"}: ticker is
    stored once at registration; label + reason drive the UI line that shows
    the user WHY, and that it was not chosen after the fact.

    large_pct = mandated large-cap floor; smid_pct = mandated small+micro
    allowance (micro is always 0). Thresholds per R&B Ch. 25 / the spec:
      large_pct >= 60  -> Nifty 50            (a majority-large mandate)
      smid_pct  >= 40  -> Nifty Smallcap 250
      otherwise        -> Nifty Midcap 150    (the typical Kordent portfolio)
    """
    alloc = (ips_policy or {}).get("allocation_policy") or {}
    large_pct = float(alloc.get("large_cap_min_pct", 30) or 0)
    smid_pct = float(alloc.get("small_cap_max_pct", 25) or 0)  # + micro (always 0)

    if large_pct >= 60:
        key = "nifty50"
        reason = f"your IPS mandates a {large_pct:.0f}% large-cap floor"
    elif smid_pct >= 40:
        key = "smallcap250"
        reason = f"your IPS allows up to {smid_pct:.0f}% small-cap"
    else:
        key = "midcap150"
        reason = (f"a mid-cap-tilted mandate (large-cap floor {large_pct:.0f}%, "
                  f"small-cap allowance {smid_pct:.0f}%)")

    b = BENCHMARKS[key]
    return {"ticker": b["ticker"], "label": b["label"], "reason": reason}

def describe_benchmark(port: dict) -> dict:
    """UI helper: label + reason for a portfolio's STORED benchmark ticker.
    Read-only — the label comes from the frozen ticker (authoritative), the
    reason is re-derived from the frozen IPS allocation for display. If the
    stored ticker and the IPS-implied ticker disagree (legacy rows backfilled
    to NIFTYBEES before per-mandate benchmarking), we say "locked at
    registration" rather than print a cap-tilt claim that contradicts the label.
    """
    ticker = (port or {}).get("benchmark_ticker") or "NIFTYBEES.NS"
    label = next((v["label"] for v in BENCHMARKS.values() if v["ticker"] == ticker), ticker)
    ips = (port.get("portfolio_profile") or {}).get("ips_policy")
    rec = choose_benchmark(ips)
    reason = rec["reason"] if rec["ticker"] == ticker else "locked at registration"
    return {"ticker": ticker, "label": label, "reason": reason}
