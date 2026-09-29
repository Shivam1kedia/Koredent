"""
yf_bse_map_probe.py — READ-ONLY dry run of the BSE symbol mapping, before any
pipeline edit. Run on Actions (bse-probe.yml), ~4 minutes.

Established 2026-09-24 (yf_bse_probe.py): Yahoo no longer recognises NUMERIC
BSE codes — 511066.BO, and even 500325.BO (Reliance) and 532540.BO (TCS), come
back quoteType NONE with no name, exchange or price. The same companies still
exist under NAME-based symbols: NAGREEKCAP.BO returned a full year of prices.

The BSE security list the pipeline already downloads (the GitHub mirror) carries
that name in its `symbol` column, beside the numeric `security_code`. The
proposed fix keeps the numeric code as identity and uses `<symbol>.BO` only for
Yahoo calls. This probe measures, before anything is changed:

  1. COVERAGE  — what share of today's BSE-only universe has a mirror symbol
  2. HIT RATE  — of a stratified sample, how many mapped symbols return prices
  3. DEPTH     — how much history they return (the attribution work needs a year)
  4. IDENTITY  — does Yahoo's name for the mapped symbol match the company
  5. SPECIAL CHARACTERS — symbols like ARE&M: as-is, or does Yahoo want a variant
"""
import random
import re
import time

import pandas as pd
import yfinance as yf

MIRROR = "https://raw.githubusercontent.com/RuchiTanmay/bseindia/main/bseindia/bse_security_list.csv"
SEED = 20260924

m = pd.read_csv(MIRROR)
m.columns = [str(c).strip().lower() for c in m.columns]
print("mirror columns:", list(m.columns), "| rows:", len(m))
m["security_code"] = m["security_code"].astype(str).str.strip()
m["symbol"] = m["symbol"].astype(str).str.strip()
code_to_sym = dict(zip(m["security_code"], m["symbol"]))
code_to_name = dict(zip(m["security_code"], m["security_name"].astype(str)))

u = pd.read_csv("universe_scored.csv", low_memory=False)
bse = u[u["ticker"].astype(str).str.endswith(".BO")].copy()
bse["code"] = bse["ticker"].str.replace(".BO", "", regex=False)
bse["symbol"] = bse["code"].map(code_to_sym)
bse["market_cap"] = pd.to_numeric(bse["market_cap"], errors="coerce")
print(f"\n[1] COVERAGE: {bse['symbol'].notna().sum()} / {len(bse)} BSE-only universe "
      f"tickers have a mirror symbol "
      f"({bse['symbol'].notna().mean():.1%})")
big = bse[bse["market_cap"] >= 100e7]
print(f"    of those >= Rs 100 Cr (the attribution universe): "
      f"{big['symbol'].notna().sum()} / {len(big)}")

have = bse[bse["symbol"].notna()].sort_values("market_cap", ascending=False)
rng = random.Random(SEED)
sample = pd.concat([
    have.head(20),                                             # largest
    have.iloc[len(have) // 2 - 10: len(have) // 2 + 10],        # middle
    have.sample(20, random_state=SEED),                        # random
]).drop_duplicates("ticker")
special = have[have["symbol"].str.contains(r"[^A-Z0-9]", regex=True)].head(10)
print(f"    sample: {len(sample)} stratified + {len(special)} with special characters")


def variants(sym):
    out = [sym]
    if re.search(r"[^A-Z0-9]", sym):
        out += [sym.replace("&", "_"), re.sub(r"[^A-Z0-9]", "", sym),
                sym.replace("&", "%26")]
    return list(dict.fromkeys(out))


def test(row):
    res = {"ticker": row["ticker"], "symbol": row["symbol"],
           "mirror_name": code_to_name.get(row["code"], "")[:28]}
    for v in variants(row["symbol"]):
        try:
            h = yf.download(f"{v}.BO", start="2021-01-01", progress=False,
                            auto_adjust=True, threads=False)
            n = len(h)
        except Exception:
            n = 0
        if n:
            res.update(hit=v, rows=n, first=h.index.min().date())
            try:
                res["yahoo_name"] = (yf.Ticker(f"{v}.BO").info or {}).get("shortName", "")[:28]
            except Exception:
                res["yahoo_name"] = "?"
            break
        time.sleep(0.8)
    else:
        res.update(hit=None, rows=0, first=None, yahoo_name="")
    time.sleep(0.8)
    return res


rows = [test(r) for _, r in sample.iterrows()]
df = pd.DataFrame(rows)
hit = df["rows"] > 0
print(f"\n[2] HIT RATE: {hit.sum()} / {len(df)} mapped symbols returned prices ({hit.mean():.0%})")
print(f"[3] DEPTH: median {df.loc[hit, 'rows'].median():.0f} days; "
      f">= 250 days (a year): {(df['rows'] >= 250).sum()} / {len(df)}")

print("\n[4] IDENTITY — mirror name vs Yahoo's name for the mapped symbol:")
with pd.option_context("display.width", 200, "display.max_rows", 100):
    print(df[["ticker", "symbol", "hit", "rows", "first", "mirror_name", "yahoo_name"]]
          .to_string(index=False))

srows = [test(r) for _, r in special.iterrows()]
if srows:
    s = pd.DataFrame(srows)
    print("\n[5] SPECIAL CHARACTERS — which variant Yahoo accepted:")
    print(s[["symbol", "hit", "rows", "mirror_name", "yahoo_name"]].to_string(index=False))

print("\nDone.")
