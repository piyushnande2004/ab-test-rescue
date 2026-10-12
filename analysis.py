"""
Reusable statistical checks for A/B test validation.
Every function takes a DataFrame with a 'group' column ('control'/'treatment').
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def split(df):
    return df[df.group == "control"], df[df.group == "treatment"]


# ---------- 1. Sample Ratio Mismatch ----------
def srm_test(df, expected_treatment_share=0.5):
    n_c = int((df.group == "control").sum())
    n_t = int((df.group == "treatment").sum())
    total = n_c + n_t
    chi2, p = stats.chisquare([n_c, n_t],
                              [total * (1 - expected_treatment_share), total * expected_treatment_share])
    return dict(n_control=n_c, n_treatment=n_t, treatment_share=n_t / total,
                chi2=chi2, p_value=p, passed=p >= 0.001)


# ---------- 2. Pre-experiment balance ----------
def balance_check(df, covariate="pre_spend"):
    c, t = split(df)
    smd = (t[covariate].mean() - c[covariate].mean()) / np.sqrt((c[covariate].var() + t[covariate].var()) / 2)
    p = stats.ttest_ind(t[covariate], c[covariate], equal_var=False).pvalue
    return dict(covariate=covariate, control_mean=c[covariate].mean(), treatment_mean=t[covariate].mean(),
                std_mean_diff=smd, p_value=p, passed=(p >= 0.01) and (abs(smd) < 0.05))


# ---------- 3. Effect tests ----------
def conversion_test(c, t):
    """Two-proportion z-test. c, t are 0/1 Series."""
    nc, nt, pc, pt = len(c), len(t), c.mean(), t.mean()
    diff = pt - pc
    se = np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
    pool = (c.sum() + t.sum()) / (nc + nt)
    z = diff / np.sqrt(pool * (1 - pool) * (1 / nc + 1 / nt))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return dict(control=pc, treatment=pt, abs_diff=diff, rel_lift=diff / pc,
                ci_low=(diff - 1.96 * se) / pc, ci_high=(diff + 1.96 * se) / pc, p_value=p)


def mean_test(c, t):
    """Welch t-test for continuous metrics. Lift is relative to the control mean."""
    diff = t.mean() - c.mean()
    se = np.sqrt(c.var() / len(c) + t.var() / len(t))
    p = stats.ttest_ind(t, c, equal_var=False).pvalue
    return dict(control=c.mean(), treatment=t.mean(), abs_diff=diff, rel_lift=diff / c.mean(),
                ci_low=(diff - 1.96 * se) / c.mean(), ci_high=(diff + 1.96 * se) / c.mean(),
                p_value=p, abs_ci_low=diff - 1.96 * se, abs_ci_high=diff + 1.96 * se)


# ---------- 4. Variance reduction & robustness ----------
def cuped_adjust(df, metric, covariate="pre_spend"):
    """CUPED: remove the part of the metric explained by pre-experiment behaviour."""
    x, y = df[covariate].values, df[metric].values
    theta = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
    return pd.Series(y - theta * (x - x.mean()), index=df.index), theta


def winsorize(series, upper_q=0.999):
    return series.clip(upper=series.quantile(upper_q))


def bootstrap_rel_lift(c, t, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    c, t = c.values, t.values
    lifts = np.empty(n_boot)
    for i in range(n_boot):
        lifts[i] = rng.choice(t, len(t)).mean() / rng.choice(c, len(c)).mean() - 1
    return tuple(np.percentile(lifts, [2.5, 97.5]))


# ---------- 5. Multiple testing ----------
def bh_correct(p_values, alpha=0.05):
    reject, p_adj, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return reject, p_adj


# ---------- 6. Time-based checks ----------
def daily_effect(df):
    rows = []
    for d in sorted(df.day.unique()):
        c, t = split(df[df.day == d])
        r = conversion_test(c.converted, t.converted)
        rows.append(dict(day=d, **r))
    return pd.DataFrame(rows)


def peeking_table(df):
    """What p-value would you have seen if you checked at the end of each day?"""
    rows = []
    for d in sorted(df.day.unique()):
        c, t = split(df[df.day <= d])
        r = conversion_test(c.converted, t.converted)
        rows.append(dict(day=d, rel_lift=r["rel_lift"], p_value=r["p_value"]))
    return pd.DataFrame(rows)


def novelty_check(df, window=3):
    days = sorted(df.day.unique())
    early = conversion_test(*[x.converted for x in split(df[df.day.isin(days[:window])])])
    late = conversion_test(*[x.converted for x in split(df[df.day.isin(days[-window:])])])
    flagged = (early["p_value"] < 0.05 and early["rel_lift"] > 0) and \
              (late["ci_low"] <= 0 or late["rel_lift"] < 0.5 * early["rel_lift"])
    return dict(early_lift=early["rel_lift"], late_lift=late["rel_lift"],
                late_ci=(late["ci_low"], late["ci_high"]), flagged=flagged)


# ---------- 7. Segments (Simpson's paradox) ----------
def segment_analysis(df, cols=("device", "is_new", "region")):
    rows = []
    for col in cols:
        for val in sorted(df[col].unique()):
            c, t = split(df[df[col] == val])
            r = conversion_test(c.converted, t.converted)
            rows.append(dict(segment=f"{col}={val}", n=len(c) + len(t), **r))
    out = pd.DataFrame(rows)
    out["significant_bh"], out["p_adj"] = bh_correct(out.p_value.values)
    return out


# ---------- 8. Bayesian view ----------
def bayesian_conversion(c, t, draws=200_000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.beta(1 + c.sum(), 1 + len(c) - c.sum(), draws)
    b = rng.beta(1 + t.sum(), 1 + len(t) - t.sum(), draws)
    return dict(prob_treatment_better=(b > a).mean(), expected_rel_lift=(b / a - 1).mean(),
                expected_loss=np.maximum(a - b, 0).mean())


# ---------- 9. Power analysis ----------
def required_sample_size(base_rate, mde_rel, alpha=0.05, power=0.8):
    es = proportion_effectsize(base_rate * (1 + mde_rel), base_rate)
    return int(np.ceil(NormalIndPower().solve_power(es, alpha=alpha, power=power, ratio=1)))


def conf(p):
    """Confidence (0-100%) that the result is not just luck = (1 - p) * 100."""
    c = (1 - p) * 100
    return ">99.99%" if c > 99.99 else f"{c:.2f}%"
