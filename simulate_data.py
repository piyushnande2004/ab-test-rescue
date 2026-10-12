"""
Step 1: Simulate a MESSY A/B test with known, planted flaws.

Planted flaws (you know the ground truth, so you can prove your checks catch them):
  1. Sample Ratio Mismatch  - treatment silently loses low-engagement mobile users
  2. Imbalanced groups      - consequence of (1): treatment has higher pre-experiment spend
  3. Novelty effect         - desktop lift decays day by day
  4. Hidden segment loss    - mobile users convert slightly worse in treatment
  5. Outlier orders         - a few giant orders inflate treatment revenue
  6. Multiple metrics       - one null metric (pages_per_session) to test false positives
"""
import numpy as np
import pandas as pd
from pathlib import Path


def simulate(n=40000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"user_id": np.arange(1, n + 1)})
    df["group"] = rng.choice(["control", "treatment"], n)          # intended 50/50
    df["day"] = rng.integers(1, 10, n)                              # 9-day test
    df["device"] = rng.choice(["mobile", "desktop"], n, p=[0.55, 0.45])
    df["is_new"] = rng.choice([1, 0], n, p=[0.4, 0.6])
    df["region"] = rng.choice(["North", "South", "East", "West"], n, p=[0.3, 0.3, 0.2, 0.2])

    z = rng.normal(size=n)                                          # hidden engagement
    df["pre_spend"] = np.exp(3 + 0.8 * z + 0.3 * rng.normal(size=n)).round(2)

    base = (np.where(df.device == "mobile", 0.08, 0.12)
            * np.where(df.is_new == 1, 0.8, 1.0)
            * np.exp(0.12 * z))
    decay = np.exp(-(df.day.values - 1) / 3)                        # novelty effect
    effect = np.where(df.device == "desktop", 0.28 * decay, -0.10)  # relative effect
    effect = np.where(df.group == "treatment", effect, 0.0)
    p = np.clip(base * (1 + effect), 0, 0.95)

    df["converted"] = (rng.random(n) < p).astype(int)
    aov = np.exp(4 + 0.3 * z + 0.4 * rng.normal(size=n))
    df["revenue"] = np.where(df.converted == 1, aov, 0.0).round(2)
    df["pages_per_session"] = rng.poisson(5, n)                     # no true effect

    # FLAW 1: bug drops ~25% of low-engagement mobile users from treatment
    drop = ((df.group == "treatment") & (df.device == "mobile")
            & (z < -0.5) & (rng.random(n) < 0.25))
    df = df[~drop].copy()

    # FLAW 5: a handful of huge orders in treatment
    idx = rng.choice(df.index[df.group == "treatment"], 5, replace=False)
    df.loc[idx, "converted"] = 1
    df.loc[idx, "revenue"] = rng.uniform(1000, 1500, 5).round(2)

    return df.reset_index(drop=True)


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    data = simulate()
    data.to_csv("data/experiment.csv", index=False)
    print(f"Saved data/experiment.csv with {len(data):,} rows")
    print(data.group.value_counts())
