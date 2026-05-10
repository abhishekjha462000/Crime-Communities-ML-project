"""
streamlit_app.py
Communities & Crime — predictive model demo
Master's project, University of Luxembourg
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# CRITICAL: import classes BEFORE joblib.load so unpickling resolves them
from pipeline_components import LemasIndicatorAdder, FeatureDropper

# ============================================================
# LOAD ARTIFACTS (cached; loaded once per session)
# ============================================================
@st.cache_resource
def load_artifacts():
    base = Path('artifacts')
    return {
        'spline':         joblib.load(base / 'production_spline.joblib'),
        'spline_no_race': joblib.load(base / 'production_spline_no_race.joblib'),
        'q_low':          joblib.load(base / 'production_q_low.joblib'),
        'q_high':         joblib.load(base / 'production_q_high.joblib'),
        'metadata':       json.loads((base / 'production_metadata.json').read_text()),
        'feature_stats':  pd.read_csv(base / 'feature_stats.csv', index_col=0),
        'interventions':  json.loads((base / 'interventions.json').read_text()),
        'samples':        json.loads((base / 'sample_communities.json').read_text()),
    }

artifacts  = load_artifacts()
metadata   = artifacts['metadata']
correction = metadata['cqr']['correction']
features   = metadata['feature_columns']
fs         = artifacts['feature_stats']

# ============================================================
# HELPERS
# ============================================================
def parse_value(v):
    """sample_communities.json stores NaNs as 'NA' strings — convert back."""
    if v is None or v == 'NA' or v == 'null':
        return np.nan
    try:
        return float(v)
    except (ValueError, TypeError):
        return np.nan

def get_initial_values(sample_key):
    """Return (values_dict, actual_crime_or_None) for the chosen sample."""
    if sample_key is None:
        values = {}
        for col in features:
            v = fs.loc[col, '50%'] if col in fs.index else np.nan
            values[col] = float(v) if not pd.isna(v) else np.nan
        return values, None
    sample = artifacts['samples'][sample_key]
    actual = sample.get('_actual_violent_crime')
    values = {col: parse_value(sample.get(col)) for col in features}
    return values, actual

def predict(X_df, model):
    return float(np.clip(model.predict(X_df)[0], 0, 1))

def cqr_interval(X_df):
    lo = artifacts['q_low'].predict(X_df)[0]  - correction
    hi = artifacts['q_high'].predict(X_df)[0] + correction
    return float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1))

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Communities & Crime Predictor",
    page_icon="📊",
    layout="wide",
)

# ============================================================
# HEADER + FAIRNESS BANNER
# ============================================================
st.title("Communities & Crime — Predictive Model")
st.caption(
    "Master's project · University of Luxembourg · "
    "UCI Communities & Crime dataset (1990–1995 US data)"
)

st.warning(
    "**For academic analysis only — not for deployment.** This model has documented "
    "fairness limitations. On held-out test data, prediction error is **2.49× higher** for "
    "communities in the top quartile of `racepctblack`. Predictions are *associative*, "
    "not causal. See **Methodology & Caveats** below before interpreting results."
)

# ============================================================
# SIDEBAR — model choice + sample picker
# ============================================================
with st.sidebar:
    st.header("Configuration")

    model_choice = st.radio(
        "Model variant",
        ["With race variables (production)", "Without race (sensitivity)"],
        help=(
            "The 'no-race' variant excludes racepctblack/racePctWhite/racePctAsian/racePctHisp. "
            "Test R² drops from 0.640 to 0.629."
        ),
    )

    st.divider()
    st.subheader("Load a sample community")
    sample_choice = st.selectbox(
        "Pick an example",
        ["(use median values)",
         "Low crime (10th percentile)",
         "Median",
         "High crime (90th percentile)"],
    )
    sample_map = {
        "(use median values)":          None,
        "Low crime (10th percentile)":  'low_crime',
        "Median":                       'median',
        "High crime (90th percentile)": 'high_crime',
    }
    sample_key = sample_map[sample_choice]

    st.divider()
    tr = metadata['test_results']
    st.markdown(
        "**About this model**\n\n"
        f"- Trained on {metadata['training_size']} communities\n"
        f"- Tested on {metadata['test_size']} held-out communities\n"
        f"- Test R² (Spline-Ridge): **{tr['spline_R2']}**\n"
        f"- Test MAE (Spline-Ridge): **{tr['spline_MAE']}**\n"
        f"- CQR coverage on test: **{metadata['cqr']['test_coverage']*100:.1f}%**"
    )

active_model = (
    artifacts['spline_no_race']
    if 'Without' in model_choice
    else artifacts['spline']
)

# ============================================================
# 1. COMMUNITY PROFILE
# ============================================================
st.header("1. Community Profile")

initial_values, actual_crime = get_initial_values(sample_key)

st.markdown(
    "Adjust the **top 10 most predictive features** below. The remaining 112 features "
    "use the selected sample's values (or median if no sample is loaded)."
)

top_10 = [item['feature'] for item in metadata['top_25_features'][:10]]
adjusted = dict(initial_values)

cols = st.columns(2)
for i, feat in enumerate(top_10):
    with cols[i % 2]:
        if feat in fs.index:
            row = fs.loc[feat]
            mn, mx = float(row['min']), float(row['max'])
            cur = initial_values.get(feat)
            if pd.isna(cur):
                cur = float(row['50%'])
            cur = max(mn, min(mx, float(cur)))
            adjusted[feat] = st.slider(
                feat,
                min_value=mn, max_value=mx, value=cur, step=0.01,
                help=f"5th–95th pct in dev: [{row['5%']:.2f}, {row['95%']:.2f}]",
            )

with st.expander("Other features (112 columns) — using sample/median values"):
    other = [f for f in features if f not in top_10]
    other_df = pd.DataFrame({
        'feature': other,
        'value': [adjusted.get(f) for f in other],
    })
    st.dataframe(other_df, height=300, use_container_width=True)

# Build prediction input — coerce to numeric (handles NaN cleanly)
X_input = pd.DataFrame([adjusted])[features].apply(pd.to_numeric, errors='coerce')

# ============================================================
# 2. PREDICTION
# ============================================================
st.header("2. Prediction")

y_pred = predict(X_input, active_model)
q_low, q_high = cqr_interval(X_input)
race_value = adjusted.get('racepctblack', 0)
if pd.isna(race_value):
    race_value = 0
q4_threshold = metadata['q4_threshold_racepctblack']
is_q4 = race_value >= q4_threshold

c1, c2, c3 = st.columns(3)
with c1:
    st.metric(
        "Point prediction",
        f"{y_pred:.3f}",
        help="Predicted ViolentCrimesPerPop, scale 0–1.",
    )
with c2:
    st.metric(
        "90% prediction interval",
        f"[{q_low:.3f}, {q_high:.3f}]",
        delta=f"width {q_high - q_low:.3f}",
        delta_color="off",
        help="CQR-calibrated; test coverage was 92.5%.",
    )
with c3:
    if is_q4:
        st.error(
            f"**Q4 community**\n\n"
            f"`racepctblack` = {race_value:.3f} ≥ {q4_threshold:.3f}\n\n"
            "Model has 2.49× higher MAE in this stratum. "
            "Predictions for Q4 communities should be interpreted with extra caution."
        )
    else:
        st.success(
            f"**Not Q4**\n\n"
            f"`racepctblack` = {race_value:.3f} < {q4_threshold:.3f}\n\n"
            "Model is more accurate in this stratum."
        )

if actual_crime is not None:
    st.caption(
        f"For reference — this sample's **actual** ViolentCrimesPerPop: **{actual_crime:.3f}**. "
        "(Predictions can differ from the actual even when features match exactly, because the "
        "model's regularized fit smooths noise.)"
    )

# ============================================================
# 3. COUNTERFACTUAL SCENARIOS
# ============================================================
st.header("3. Counterfactual Scenarios")

st.markdown(
    "What does the model predict if one feature shifts? **These are predicted associations, "
    "not causal effects.** Per-community Δp values are typically **smaller than the prediction "
    "interval width**, so individual numbers should not be over-interpreted."
)

cf_records = []
for intv in artifacts['interventions']:
    feat, delta, frame = intv['feature'], intv['delta'], intv['frame']
    X_cf = X_input.copy()
    if feat in X_cf.columns:
        X_cf[feat] = (X_cf[feat] + delta).clip(0, 1)
    y_cf = predict(X_cf, active_model)
    cf_records.append({
        'Intervention':     frame,
        'Feature':          feat,
        'Δ feature':        f"{delta:+.2f}",
        'Predicted crime':  f"{y_cf:.3f}",
        'Δ prediction':     f"{y_cf - y_pred:+.4f}",
    })

st.dataframe(pd.DataFrame(cf_records), hide_index=True, use_container_width=True)

st.caption(
    f"For context: the prediction interval width here is **{q_high - q_low:.3f}**. "
    "The largest intervention Δ in this list is far smaller than that. This is the "
    "inherent honesty of the model: per-community effects are noisy. Aggregate patterns "
    "across many communities are more reliable than any individual number."
)

# ============================================================
# 4. METHODOLOGY & CAVEATS
# ============================================================
st.header("4. Methodology & Caveats")

with st.expander("Test set performance (held-out 199 communities)"):
    tr = metadata['test_results']
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Spline-Ridge (deployed)**")
        st.markdown(f"- Test R²: **{tr['spline_R2']}**")
        st.markdown(f"- Test MAE: {tr['spline_MAE']}")
    with c2:
        st.markdown("**Stacked ensemble (reference)**")
        st.markdown(f"- Test R²: **{tr['stack_R2']}**")
        st.markdown(f"- Test MAE: {tr['stack_MAE']}")
        st.markdown(f"- Test RMSE: {tr['stack_RMSE']}")

    st.divider()
    st.markdown("**Stratified test performance (the fairness gap)**")
    st.markdown(f"- Q4 (n=50): R² = {tr['q4_R2']}, MAE = {tr['q4_MAE']}")
    st.markdown(f"- not-Q4 (n=149): R² = {tr['notq4_R2']}, MAE = {tr['notq4_MAE']}")
    st.markdown(f"- **MAE ratio Q4 / not-Q4: {tr['q4_mae_ratio']}×**")

with st.expander("Caveats"):
    for c in metadata['caveats']:
        st.markdown(f"- {c}")

with st.expander("Conformalized Quantile Regression (CQR) — how the intervals work"):
    cqr = metadata['cqr']
    st.markdown(
        f"Prediction intervals come from **Conformalized Quantile Regression** "
        f"(Romano, Patterson & Candès, 2019). Two LightGBM models predict the "
        f"{cqr['alpha']/2*100:.0f}th and {(1-cqr['alpha']/2)*100:.0f}th conditional quantiles. "
        f"A conformal correction (here **{cqr['correction']:.4f}**) widens the bounds "
        f"to achieve nominal {cqr['target_coverage']*100:.0f}% coverage with a finite-sample "
        "guarantee.\n\n"
        f"- Dev OOF coverage: {cqr['dev_oof_coverage']*100:.1f}%\n"
        f"- Test coverage: **{cqr['test_coverage']*100:.1f}%**\n"
        f"- Mean interval width: {cqr['mean_width']:.3f} (target scale 0–1)"
    )

with st.expander("Top 25 features by permutation importance"):
    st.dataframe(
        pd.DataFrame(metadata['top_25_features']),
        hide_index=True, use_container_width=True,
    )

st.divider()
st.caption(
    f"Production model: **{metadata['recommended_live_model']}** · "
    f"Trained on **{metadata['training_size']}** dev communities · "
    f"Validated on **{metadata['test_size']}** held-out test communities."
)