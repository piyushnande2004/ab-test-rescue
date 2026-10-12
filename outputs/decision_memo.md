# Decision Memo: New Checkout Page A/B Test

## Recommendation: DO NOT SHIP. The experiment is invalid. Fix assignment, then re-run.

### What the PM saw
- Conversion: 8.78% -> 9.59% (+9.3%, confidence = 99.46%)
- Revenue per user (raw): +13.0% (confidence = 99.65%)

### Why we cannot trust it
- Sample Ratio Mismatch (treatment share 48.9%, p = 2.3e-05); randomization is broken.
- Groups differ before the test (pre-spend SMD = 0.03); treatment and control are not comparable at the start.
- Novelty effect: lift fell from +16.0% (first 3 days) to +8.4% (last 3 days).
- Peeking risk: p < 0.05 first appeared on day 3, before the test ended.
- Revenue lift is not distinguishable from zero once outliers and CUPED are handled (+7.1%, CI -0.2% to +14.4%).

### Corrected numbers
- Revenue per user after winsorizing + CUPED: +7.1% (95% CI -0.2% to +14.4%); bootstrap CI on winsorized lift: +1.2% to +16.6%
- CUPED variance reduction on revenue: CI width 17.44 pp -> 14.66 pp
- Bayesian: P(treatment better on conversion) = 99.7%
- Estimated monthly revenue impact at 500,000 users: ₹202,720 (range ₹-7,046 to ₹412,486)

### Segment results (conversion lift)
| segment        |     n |   rel_lift |   p_adj | significant_bh   |
|:---------------|------:|-----------:|--------:|:-----------------|
| device=desktop | 18071 |     0.1497 |  0.0052 | True             |
| device=mobile  | 21073 |    -0.0041 |  0.9343 | False            |
| is_new=0       | 23415 |     0.0801 |  0.0971 | False            |
| is_new=1       | 15729 |     0.1151 |  0.0971 | False            |
| region=East    |  7818 |     0.0187 |  0.9125 | False            |
| region=North   | 11777 |     0.0749 |  0.2846 | False            |
| region=South   | 11657 |     0.08   |  0.2846 | False            |
| region=West    |  7892 |     0.2139 |  0.023  | True             |

### Metrics with Benjamini-Hochberg correction
| metric                 |   control |   treatment |   rel_lift |   p_value | significant_raw   | significant_bh   |   p_adj |
|:-----------------------|----------:|------------:|-----------:|----------:|:------------------|:-----------------|--------:|
| conversion_rate        |    0.0878 |      0.0959 |     0.0925 |    0.0054 | True              | True             |  0.0108 |
| revenue_per_user (raw) |    5.7169 |      6.4604 |     0.1301 |    0.0035 | True              | True             |  0.0108 |
| avg_order_value        |   65.0008 |     67.3175 |     0.0356 |    0.1806 | False             | False            |  0.2408 |
| pages_per_session      |    4.9756 |      4.9989 |     0.0047 |    0.2998 | False             | False            |  0.2998 |

### Next steps
1. Fix the assignment bug that is dropping mobile users from treatment.
2. Re-run for at least 183,681 users per group (to detect a 3% relative lift at 80% power); pre-register metrics and stopping rule.
3. Segment view: desktop lift +15.0% vs mobile -0.4%. Consider a desktop-only test; mobile shows no clear benefit.

### Limitations
- Simulated data; real data would need bot filtering and user-level deduplication.
- CUPED assumes pre-experiment spend is unaffected by treatment (true by construction).
- Monthly revenue impact assumes the effect persists and traffic stays constant.