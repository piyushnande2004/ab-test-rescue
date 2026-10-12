"""
Step 2: Run the full audit -> prints findings, saves charts to outputs/, writes outputs/decision_memo.md
Usage:  python run_report.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis as an
from simulate_data import simulate

sys.stdout.reconfigure(encoding="utf-8")

MONTHLY_USERS = 500_000          # <- change: users who would see the new checkout each month
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
DATA = Path("data/experiment.csv")

if not DATA.exists():
    Path("data").mkdir(exist_ok=True)
    simulate().to_csv(DATA, index=False)
df = pd.read_csv(DATA)
c, t = an.split(df)

# ---- 1-2. Validity checks
srm = an.srm_test(df)
bal = an.balance_check(df)

# ---- 3. Headline (what the PM saw) vs. corrected metrics
naive_conv = an.conversion_test(c.converted, t.converted)
naive_rev = an.mean_test(c.revenue, t.revenue)

df["rev_w"] = an.winsorize(df.revenue)
df["rev_w_cuped"], theta = an.cuped_adjust(df, "rev_w")
df["conv_cuped"], _ = an.cuped_adjust(df, "converted")
c, t = an.split(df)

rev_w = an.mean_test(c.rev_w, t.rev_w)
rev_wc = an.mean_test(c.rev_w_cuped, t.rev_w_cuped)
conv_cuped = an.mean_test(c.conv_cuped, t.conv_cuped)
boot_lo, boot_hi = an.bootstrap_rel_lift(c.rev_w, t.rev_w)

# ---- 4. Multiple metrics + BH correction
aov_c, aov_t = c[c.converted == 1].revenue, t[t.converted == 1].revenue
metrics = pd.DataFrame([
    dict(metric="conversion_rate", **naive_conv),
    dict(metric="revenue_per_user (raw)", **naive_rev),
    dict(metric="avg_order_value", **an.mean_test(an.winsorize(aov_c), an.winsorize(aov_t))),
    dict(metric="pages_per_session", **an.mean_test(c.pages_per_session, t.pages_per_session)),
])[["metric", "control", "treatment", "rel_lift", "p_value"]]
metrics["significant_raw"] = metrics.p_value < 0.05
metrics["significant_bh"], metrics["p_adj"] = an.bh_correct(metrics.p_value.values)

# ---- 5. Time, segments, Bayesian, power
daily = an.daily_effect(df)
peek = an.peeking_table(df)
novelty = an.novelty_check(df)
segs = an.segment_analysis(df)
bayes = an.bayesian_conversion(c.converted, t.converted)
need_n = an.required_sample_size(c.converted.mean(), 0.03)
first_sig = peek[peek.p_value < 0.05].day.min()

harmful = segs[(segs.significant_bh) & (segs.rel_lift < 0)]
money_mid = rev_wc["abs_diff"] * MONTHLY_USERS
money_lo, money_hi = rev_wc["abs_ci_low"] * MONTHLY_USERS, rev_wc["abs_ci_high"] * MONTHLY_USERS

# ---- Charts
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(["control", "treatment"], [srm["n_control"], srm["n_treatment"]], color=["#6b7280", "#2563eb"])
ax.axhline((srm["n_control"] + srm["n_treatment"]) / 2, ls="--", c="red", label="expected (50/50)")
ax.set_title(f"Sample Ratio Mismatch  (confidence = {an.conf(srm['p_value'])})"); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "1_srm.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(daily.day, daily.rel_lift * 100,
            yerr=[(daily.rel_lift - daily.ci_low) * 100, (daily.ci_high - daily.rel_lift) * 100],
            fmt="o-", capsize=4)
ax.axhline(0, c="red", ls="--"); ax.set_xlabel("Exposure day"); ax.set_ylabel("Conversion lift (%)")
ax.set_title("Novelty effect: lift fades over time"); fig.tight_layout()
fig.savefig(OUT / "2_novelty.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(peek.day, peek.p_value, "o-"); ax.axhline(0.05, c="red", ls="--", label="p = 0.05")
ax.set_yscale("log"); ax.set_xlabel("Day you stopped"); ax.set_ylabel("p-value (log)")
ax.set_title("Peeking: p-value if you stopped on each day"); ax.legend(); fig.tight_layout()
fig.savefig(OUT / "3_peeking.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
s = segs.sort_values("rel_lift")
colors = ["#dc2626" if v < 0 else "#16a34a" for v in s.rel_lift]
ax.barh(s.segment, s.rel_lift * 100, xerr=[(s.rel_lift - s.ci_low) * 100, (s.ci_high - s.rel_lift) * 100],
        color=colors, capsize=3)
ax.axvline(0, c="black"); ax.set_xlabel("Conversion lift (%)")
ax.set_title("Segments: who wins, who loses"); fig.tight_layout()
fig.savefig(OUT / "4_segments.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 3.5))
rows = [("Raw revenue/user", naive_rev), ("Winsorized", rev_w), ("Winsorized + CUPED", rev_wc)]
for i, (lab, r) in enumerate(rows):
    ax.errorbar(r["rel_lift"] * 100, i, xerr=[[(r["rel_lift"] - r["ci_low"]) * 100], [(r["ci_high"] - r["rel_lift"]) * 100]],
                fmt="o", capsize=5)
ax.set_yticks(range(3)); ax.set_yticklabels([r[0] for r in rows]); ax.axvline(0, c="red", ls="--")
ax.set_xlabel("Revenue-per-user lift (%) with 95% CI"); ax.set_title("Outliers + CUPED change the story")
fig.tight_layout(); fig.savefig(OUT / "5_revenue_cleanup.png", dpi=150); plt.close(fig)

# ---- Decision logic
problems = []
if not srm["passed"]:
    problems.append(f"Sample Ratio Mismatch (treatment share {srm['treatment_share']:.1%}, p = {srm['p_value']:.1e}); randomization is broken.")
if not bal["passed"]:
    problems.append(f"Groups differ before the test (pre-spend SMD = {bal['std_mean_diff']:.2f}); treatment and control are not comparable at the start.")
if novelty["flagged"]:
    problems.append(f"Novelty effect: lift fell from {novelty['early_lift']:+.1%} (first 3 days) to {novelty['late_lift']:+.1%} (last 3 days).")
if len(harmful):
    problems.append("Harmful segment(s) after multiple-testing correction: " + ", ".join(harmful.segment) + ".")
if first_sig and first_sig < peek.day.max():
    problems.append(f"Peeking risk: p < 0.05 first appeared on day {first_sig}, before the test ended.")
if rev_wc["ci_low"] <= 0 <= rev_wc["ci_high"]:
    problems.append(f"Revenue lift is not distinguishable from zero once outliers and CUPED are handled ({rev_wc['rel_lift']:+.1%}, CI {rev_wc['ci_low']:+.1%} to {rev_wc['ci_high']:+.1%}).")

critical = (not srm["passed"])
if critical:
    decision = "DO NOT SHIP. The experiment is invalid. Fix assignment, then re-run."
elif problems:
    decision = "DO NOT SHIP YET. Run longer and/or re-test per segment."
elif naive_conv["p_value"] < 0.05 and naive_conv["rel_lift"] > 0:
    decision = "SHIP."
else:
    decision = "INCONCLUSIVE. No evidence of an effect."

desk = segs.loc[segs.segment == "device=desktop", "rel_lift"].iloc[0]
mob = segs.loc[segs.segment == "device=mobile", "rel_lift"].iloc[0]

# ---- Print + memo
lines = [
    "# Decision Memo: New Checkout Page A/B Test", "",
    f"## Recommendation: {decision}", "",
    "### What the PM saw",
    f"- Conversion: {naive_conv['control']:.2%} -> {naive_conv['treatment']:.2%} "
    f"({naive_conv['rel_lift']:+.1%}, confidence = {an.conf(naive_conv['p_value'])})",
    f"- Revenue per user (raw): {naive_rev['rel_lift']:+.1%} (confidence = {an.conf(naive_rev['p_value'])})", "",
    "### Why we cannot trust it",
    *[f"- {p}" for p in problems], "",
    "### Corrected numbers",
    f"- Revenue per user after winsorizing + CUPED: {rev_wc['rel_lift']:+.1%} "
    f"(95% CI {rev_wc['ci_low']:+.1%} to {rev_wc['ci_high']:+.1%}); bootstrap CI on winsorized lift: {boot_lo:+.1%} to {boot_hi:+.1%}",
    f"- CUPED variance reduction on revenue: CI width {(naive_rev['ci_high']-naive_rev['ci_low'])*100:.2f} pp -> {(rev_wc['ci_high']-rev_wc['ci_low'])*100:.2f} pp",
    f"- Bayesian: P(treatment better on conversion) = {bayes['prob_treatment_better']:.1%}",
    f"- Estimated monthly revenue impact at {MONTHLY_USERS:,} users: ₹{money_mid:,.0f} "
    f"(range ₹{money_lo:,.0f} to ₹{money_hi:,.0f})", "",
    "### Segment results (conversion lift)",
    segs[["segment", "n", "rel_lift", "p_adj", "significant_bh"]].round(4).to_markdown(index=False), "",
    "### Metrics with Benjamini-Hochberg correction",
    metrics.round(4).to_markdown(index=False), "",
    "### Next steps",
    "1. Fix the assignment bug that is dropping mobile users from treatment.",
    f"2. Re-run for at least {need_n:,} users per group (to detect a 3% relative lift at 80% power); pre-register metrics and stopping rule.",
    f"3. Segment view: desktop lift {desk:+.1%} vs mobile {mob:+.1%}. Consider a desktop-only test; mobile shows no clear benefit.", "",
    "### Limitations",
    "- Simulated data; real data would need bot filtering and user-level deduplication.",
    "- CUPED assumes pre-experiment spend is unaffected by treatment (true by construction).",
    "- Monthly revenue impact assumes the effect persists and traffic stays constant.",
]
memo = "\n".join(lines)
(OUT / "decision_memo.md").write_text(memo, encoding="utf-8")
print(memo)
print("\nCharts and memo saved in the 'outputs' folder.")
