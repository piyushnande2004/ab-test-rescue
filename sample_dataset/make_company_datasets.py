"""Creates 3 practice experiment files from imaginary companies (simulated, not real data)."""
import numpy as np
import pandas as pd
from pathlib import Path

out = Path("sample_datasets"); out.mkdir(exist_ok=True)


# 1) FoodRush (food delivery): clean test, real +8% lift -> app should say PASS and a real win
def foodrush(n=30000, seed=1):
    r = np.random.default_rng(seed)
    z = r.normal(size=n)
    d = pd.DataFrame({
        "user_id": np.arange(n),
        "variant": r.choice(["A_control", "B_promo"], n),
        "test_day": r.integers(1, 15, n),
        "past_orders": r.poisson(np.exp(1 + 0.5 * z)),
        "city": r.choice(["Nagpur", "Pune", "Mumbai", "Delhi"], n),
        "platform": r.choice(["android", "ios", "web"], n, p=[0.6, 0.25, 0.15]),
    })
    p = 0.18 * np.exp(0.1 * z) * np.where(d.variant == "B_promo", 1.08, 1.0)
    d["ordered"] = (r.random(n) < p)
    return d


# 2) StreamFlix (video streaming): broken split + novelty effect -> app should show FAIL and fading lift
def streamflix(n=40000, seed=2):
    r = np.random.default_rng(seed)
    z = r.normal(size=n)
    d = pd.DataFrame({
        "uid": np.arange(n),
        "page_version": r.choice(["old_page", "new_page"], n),
        "day_of_test": r.integers(1, 15, n),
        "prior_visits": r.poisson(np.exp(1.5 + 0.4 * z)),
        "country": r.choice(["IN", "US", "UK", "BR"], n, p=[0.4, 0.3, 0.2, 0.1]),
        "device": r.choice(["phone", "laptop", "smart_tv"], n, p=[0.5, 0.35, 0.15]),
    })
    lift = 0.30 * np.exp(-(d.day_of_test - 1) / 3)
    p = 0.10 * np.exp(0.15 * z) * (1 + np.where(d.page_version == "new_page", lift, 0))
    d["subscribed"] = (r.random(n) < p).astype(int)
    bug = (d.page_version == "new_page") & (d.device == "smart_tv") & (r.random(n) < 0.5)
    return d[~bug].reset_index(drop=True)          # smart-TV users silently lost from new_page


# 3) PayBank (loan app banner): small true effect, small sample -> app should say inconclusive
def paybank(n=12000, seed=3):
    r = np.random.default_rng(seed)
    d = pd.DataFrame({
        "customer": np.arange(n),
        "group": r.choice(["holdout", "exposed"], n),
        "age_band": r.choice(["18-25", "26-35", "36-50", "50+"], n),
        "salary_account": r.choice(["yes", "no"], n, p=[0.7, 0.3]),
    })
    p = 0.06 * np.where(d.group == "exposed", 1.03, 1.0)
    d["applied_loan"] = (r.random(n) < p).astype(int)
    return d


foodrush().to_csv(out / "foodrush_promo_test.csv", index=False)
streamflix().to_csv(out / "streamflix_signup_test.csv", index=False)
paybank().to_csv(out / "paybank_loan_banner.csv", index=False)
print("Created 3 files in sample_datasets/")
