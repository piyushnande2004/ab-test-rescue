"""
A/B Test Rescue app: upload ANY experiment CSV, map your columns, get a health report.
Run:  streamlit run app.py
Only 2 columns are required: a group column and a 0/1 outcome column.
Optional: day (novelty + peeking), pre-experiment metric (balance check), segment columns.
"""
import pandas as pd
import streamlit as st
import analysis as an

st.set_page_config(page_title="A/B Test Rescue", layout="wide")
st.title("A/B Test Rescue: Experiment Health Report")

file = st.sidebar.file_uploader("Upload experiment CSV", type="csv")
if file:
    raw = pd.read_csv(file)
else:
    st.info("No file uploaded. Showing the sample dataset.")
    raw = pd.read_csv("data/experiment.csv")

cols = list(raw.columns)
NONE = "(none)"


def pick(label, name, optional=False):
    options = ([NONE] if optional else []) + cols
    idx = options.index(name) if name in options else 0
    return st.sidebar.selectbox(label, options, index=idx)


# ---- Smart guesses so the menus start on sensible columns
def is_binary(c):
    return set(raw[c].dropna().unique()) <= {0, 1, True, False}


def first_match(words):
    for c in cols:
        if any(w in c.lower() for w in words):
            return c
    return None


def is_text(c):
    return not pd.api.types.is_numeric_dtype(raw[c]) and not pd.api.types.is_bool_dtype(raw[c])


text_two = [c for c in cols if is_text(c) and raw[c].nunique() == 2]
g_guess = text_two[0] if text_two else (first_match(["group", "variant", "version"]) or cols[0])
binary_cols = [c for c in cols if is_binary(c) and c != g_guess]
o_guess = binary_cols[-1] if binary_cols else cols[-1]
d_guess = first_match(["day"]) or NONE
v_guess = first_match(["pre", "past", "prior"]) or NONE


st.sidebar.header("Map your columns")
group_col = pick("Group column", g_guess)
values = sorted(raw[group_col].dropna().astype(str).unique())
if len(values) > 10:
    st.error(f"'{group_col}' has {len(values)} different values, so it looks like an ID column. "
             "Pick the column that says which group each user was in (e.g. control / treatment).")
    st.stop()
if len(values) < 2:
    st.error("The group column needs at least 2 different values.")
    st.stop()
CONTROL_WORDS = ("control", "holdout", "old", "baseline", "gate_30")
c_idx = next((i for i, v in enumerate(values) if any(w in v.lower() for w in CONTROL_WORDS)), 0)
control_val = st.sidebar.selectbox("Control value", values, index=c_idx)
treat_val = st.sidebar.selectbox("Treatment value", [v for v in values if v != control_val])
conv_col = pick("Outcome column (0/1 or True/False)", o_guess)
day_col = pick("Day column (optional)", d_guess, optional=True)
cov_col = pick("Pre-experiment metric (optional)", v_guess, optional=True)
seg_options = [c for c in cols if c not in (group_col, conv_col) and raw[c].nunique() <= 20]
seg_default = [c for c in seg_options if is_text(c) and raw[c].nunique() >= 2]
seg_cols = st.sidebar.multiselect("Segment columns (optional)", seg_options, default=seg_default)
share = st.sidebar.slider("Intended treatment share", 0.1, 0.9, 0.5, 0.05)

# ---- Build a standard table
df = pd.DataFrame({"group": raw[group_col].astype(str).map({control_val: "control", treat_val: "treatment"}),
                   "converted": raw[conv_col]})
if day_col != NONE:
    df["day"] = pd.to_numeric(raw[day_col], errors="coerce")
if cov_col != NONE:
    df["pre_spend"] = pd.to_numeric(raw[cov_col], errors="coerce")
for s in seg_cols:
    df[s] = raw[s]
df = df.dropna(subset=["group", "converted"])
if not is_binary(conv_col):
    st.error(f"'{conv_col}' is not a 0/1 or True/False column. Pick the column that says whether "
             "each user converted (e.g. ordered, subscribed, applied_loan).")
    st.stop()
df["converted"] = df["converted"].astype(int)
if df.groupby("group").converted.mean().min() == 0:
    st.error("One group has 0% conversion, so the wrong columns are probably selected. Check the outcome column.")
    st.stop()
df = df.reset_index(drop=True)
c, t = an.split(df)

# ---- Headline checks
srm = an.srm_test(df, share)
res = an.conversion_test(c.converted, t.converted)
col1, col2, col3 = st.columns(3)
col1.metric("Sample Ratio Check", "PASS" if srm["passed"] else "FAIL",
            f"treatment share {srm['treatment_share']:.1%}", delta_color="off")
if "pre_spend" in df:
    bal = an.balance_check(df.dropna(subset=["pre_spend"]))
    col2.metric("Pre-test Balance", "PASS" if bal["passed"] else "FAIL",
                f"SMD {bal['std_mean_diff']:.3f}", delta_color="off")
else:
    col2.metric("Pre-test Balance", "N/A", "no pre-experiment column", delta_color="off")
col3.metric("Headline conversion lift", f"{res['rel_lift']:+.1%}",
            f"confidence = {an.conf(res['p_value'])}", delta_color="off")

if not srm["passed"]:
    st.error("Sample Ratio Mismatch detected. Randomization is likely broken; do not trust the headline result.")
else:
    st.success("Group sizes match the intended split.")

st.write(f"Control: {res['control']:.2%}  |  Treatment: {res['treatment']:.2%}  |  "
         f"95% CI for relative lift: {res['ci_low']:+.1%} to {res['ci_high']:+.1%}")

# ---- Time checks
if "day" in df and df.day.notna().any():
    d = df.dropna(subset=["day"])
    st.subheader("Daily lift (novelty effect check)")
    st.line_chart(an.daily_effect(d).set_index("day")["rel_lift"])
    st.subheader("Peeking: p-value by stopping day")
    st.line_chart(an.peeking_table(d).set_index("day")["p_value"])
else:
    st.warning("No day column selected, so novelty and peeking checks are skipped.")

# ---- Segments
if seg_cols:
    st.subheader("Segment analysis (BH-corrected)")
    st.dataframe(an.segment_analysis(df, cols=seg_cols).round(4))

st.subheader("Bayesian view")
b = an.bayesian_conversion(c.converted, t.converted)
st.write(f"Probability treatment beats control: **{b['prob_treatment_better']:.1%}**  |  "
         f"Expected loss if wrong: {b['expected_loss']:.4%}")
