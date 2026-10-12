# A/B Test Rescue: Auditing a Messy Experiment

A tool that checks whether an A/B test result can be trusted before a company ships.

## The scenario
A product manager claims a new checkout page lifted conversion by **+9.3%**. I audited
the test and found the result can't be trusted.

![Health report on the sample test](C:\Users\victus\OneDrive\Desktop\AB-Test-resuce\screeshots\sample_fail.png)

## What the audit found
- **Sample Ratio Mismatch:** treatment got 48.9% of users instead of 50%, so randomization was broken.
- **Novelty effect:** the lift faded over the test.
- **Peeking risk:** the result first looked significant on day 3, before the test ended.
- **Outliers:** a few giant orders inflated revenue. Raw lift of +13% fell to +7% after cleanup, and the confidence interval included zero.
- **Segments:** desktop +15%, mobile no benefit.

**Verdict: do not ship, fix the assignment bug and re-run.**

![Segment results](C:\Users\victus\OneDrive\Desktop\AB-Test-resuce\screeshots\sample segment.png)

## Three datasets, three verdicts
| Dataset | Result | Verdict |
|---|---|---|
| Sample checkout test | Sample ratio check FAIL | Do not ship |
| StreamFlix signup test | Sample ratio check FAIL, lift fades over time | Do not trust |
| FoodRush promo test | All checks pass, +10.5% lift | Ship |
| PayBank loan banner | CI from -14.6% to +12.8% | Inconclusive |

![StreamFlix broken test](C:\Users\victus\OneDrive\Desktop\AB-Test-resuce\screeshots\stream_flix_fail.png)
![FoodRush clean test](C:\Users\victus\OneDrive\Desktop\AB-Test-resuce\screeshots\foodrush_pass.png)

## A lesson about segments
Even after multiple-testing correction, some segments look significant by chance
(Nagpur, Pune and android in FoodRush all had the same true lift). I treat segment
results as leads for a follow-up test, not conclusions.

![FoodRush segments]C:\Users\victus\OneDrive\Desktop\AB-Test-resuce\screeshots\foodrush_segments.png)

## How to run
    pip install -r requirements.txt
    python simulate_data.py
    python run_report.py
    streamlit run app.py

## Limitations
- All data is simulated, with flaws I planted so I could verify the checks catch them.
- CUPED assumes pre-experiment behaviour is unaffected by the test.
