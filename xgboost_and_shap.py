"""
XGBoost Training and SHAP Explainability Script
================================================

Execution order (run after pipeline.py has been verified):
    python xgboost_and_shap.py

What this script does:
    1. Loads and preprocesses data via pipeline.run_pipeline()
    2. Trains an XGBoost regression model to predict future cognitive score
    3. Evaluates the model against a naive baseline (overall and per subgroup)
    4. Runs GroupKFold cross-validation for robust performance estimates
    5. Computes SHAP values for every patient (test set + full cohort)
    6. Generates and saves SHAP and evaluation visualisations
    7. Labels each patient as Improving / Stable / Deteriorating
    8. Saves predictions, SHAP values, and a performance report

Outputs written to outputs/:
    predictions_test.csv       — test-set predictions and labels (evaluation only)
    predictions_all.csv        — one row per patient, all patients (used by chatbot)
    feature_importance.csv     — mean absolute SHAP per feature, all 82 features
    model_report.txt           — human-readable performance report
    shap_beeswarm.png          — SHAP beeswarm plot (top 15 features)
    shap_bar.png               — SHAP mean absolute importance bar chart (top 15)
    shap_waterfall_sample.png  — SHAP waterfall for one representative patient
    residual_analysis.png      — predicted vs actual + residual distribution
    confusion_matrix.png       — labelled confusion matrix heatmap
    shap_dependence.png        — SHAP dependence plots for top 3 actionable features
    shap_values.pkl            — SHAP values + feature names + full dataset
                                 (consumed by generate_shap_phenotype.py)
"""

import os
import numpy as np
import pandas as pd
import pickle
import textwrap

# Set non-interactive backend BEFORE importing pyplot.
# Without this the script crashes on headless servers (no display attached).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import GroupKFold
from scipy.stats import pearsonr

import xgboost as xgb
import shap

from pipeline import run_pipeline
import utils
from utils import get_label, XGB_PARAMS


# ── Constants ──────────────────────────────────────────────────────────────────

DATA_PATH  = "chronic_disease_progression.csv"
OUTPUT_DIR = "outputs"
LABEL_ORDER = ["Improving", "Stable", "Deteriorating"]


# =============================================================================
# HELPERS
# =============================================================================

def _disease_from_row(row: pd.Series) -> str:
    """Recover disease name from one-hot encoded columns."""
    for disease in ["Alzheimer's", "Diabetes", "Parkinson's"]:
        if row.get(f"Disease_{disease}", 0) == 1:
            return disease
    return "Unknown"


def _subgroup_metrics(results: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Compute MAE and label-agreement per unique value in group_col.
    Returns a DataFrame sorted by group value.
    """
    rows = []
    for group, grp in results.groupby(group_col):
        mae = mean_absolute_error(grp["actual_future_score"], grp["predicted_future_score"])
        agreement = (grp["actual_label"] == grp["predicted_label"]).mean()
        rows.append({
            group_col:   group,
            "N":         len(grp),
            "MAE":       round(mae, 3),
            "Agreement": f"{agreement:.1%}",
        })
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def cross_validate(X_all: pd.DataFrame, y_all: pd.Series, groups: pd.Series,
                   n_splits: int = 5) -> dict:
    """
    GroupKFold cross-validation — no patient leaks across folds.
    Returns mean ± std for MAE, RMSE, and R².
    """
    gkf   = GroupKFold(n_splits=n_splits)
    maes, rmses, r2s = [], [], []

    for train_idx, val_idx in gkf.split(X_all, y_all, groups=groups):
        X_tr, X_vl = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_vl = y_all.iloc[train_idx], y_all.iloc[val_idx]

        # No early stopping in CV to avoid leakage; use a fixed tree budget.
        # Exclude early_stopping_rounds and override n_estimators.
        _skip = {"early_stopping_rounds", "n_estimators"}
        m = xgb.XGBRegressor(
            **{k: v for k, v in XGB_PARAMS.items() if k not in _skip},
            n_estimators=200,
        )
        m.fit(X_tr, y_tr, verbose=False)
        p = m.predict(X_vl)

        maes.append(mean_absolute_error(y_vl, p))
        rmses.append(np.sqrt(mean_squared_error(y_vl, p)))
        r2s.append(r2_score(y_vl, p))

    return {
        "MAE_mean":  round(float(np.mean(maes)),  3),
        "MAE_std":   round(float(np.std(maes)),   3),
        "RMSE_mean": round(float(np.mean(rmses)), 3),
        "RMSE_std":  round(float(np.std(rmses)),  3),
        "R2_mean":   round(float(np.mean(r2s)),   3),
        "R2_std":    round(float(np.std(r2s)),    3),
    }


# =============================================================================
# VISUALISATIONS
# =============================================================================

def _save_residual_plot(y_true: np.ndarray, y_pred: np.ndarray, path: str) -> None:
    """Two-panel residual analysis: predicted vs actual, and residual histogram."""
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: scatter — predicted vs actual
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.4, edgecolors="none", s=20, color="steelblue")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual future cognitive score")
    ax.set_ylabel("Predicted future cognitive score")
    ax.set_title("Predicted vs Actual")
    ax.legend(fontsize=9)
    r, _ = pearsonr(y_true, y_pred)
    ax.text(0.05, 0.93, f"Pearson r = {r:.3f}", transform=ax.transAxes,
            fontsize=10, color="darkred")

    # Panel 2: residual distribution
    ax = axes[1]
    ax.hist(residuals, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Residual (predicted − actual)")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")
    ax.text(0.05, 0.93, f"Mean = {residuals.mean():.2f}\nStd = {residuals.std():.2f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top")

    fig.suptitle("Model Residual Analysis (Test Set)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_confusion_matrix(actual: pd.Series, predicted: pd.Series, path: str) -> None:
    """Save a labelled confusion matrix heatmap."""
    cm_data = confusion_matrix(actual, predicted, labels=LABEL_ORDER)
    fig, ax  = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_data, display_labels=LABEL_ORDER)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Progression Labels\n(rows = actual, columns = predicted)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_waterfall(explainer: shap.TreeExplainer, X_sample: pd.DataFrame,
                    pred_score: float, actual_score: float, path: str) -> None:
    """SHAP waterfall plot for a single representative patient.
    shap.waterfall_plot creates its own figure, so we title and save via plt.
    """
    shap_exp = explainer(X_sample)
    shap.waterfall_plot(shap_exp[0], max_display=12, show=False)
    plt.title(
        f"SHAP Waterfall — Representative Patient\n"
        f"Predicted: {pred_score:.1f}  |  Actual: {actual_score:.1f}",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# REPORT
# =============================================================================

def _build_report(
    model_mae: float,
    naive_mae: float,
    rmse: float,
    r2: float,
    agreement: float,
    pearson_r: float,
    cv: dict,
    results: pd.DataFrame,
    disease_metrics: pd.DataFrame,
    label_metrics: pd.DataFrame,
    importance_top: pd.Series,
    n_train: int,
    n_test: int,
    n_patients_train: int,
    n_patients_test: int,
    n_features: int,
) -> str:
    cm_arr = confusion_matrix(results["actual_label"], results["predicted_label"],
                              labels=LABEL_ORDER)
    cm_lines = []
    cm_lines.append(f"  {'':15s}  " + "  ".join(f"{l:>13s}" for l in LABEL_ORDER))
    for i, row_label in enumerate(LABEL_ORDER):
        cm_lines.append(f"  Actual {row_label:<10s}  " + "  ".join(f"{cm_arr[i,j]:>13d}" for j in range(3)))

    top_feat_lines = "\n".join(
        f"  {i+1:2d}. {feat:<45s} {val:.4f}"
        for i, (feat, val) in enumerate(importance_top.items())
    )
    disease_lines = disease_metrics.to_string(index=False)
    label_lines   = label_metrics.to_string(index=False)

    return textwrap.dedent(f"""\
    ╔══════════════════════════════════════════════════════════╗
    ║  MODEL PERFORMANCE REPORT — XGBoost Cognitive Prediction ║
    ╚══════════════════════════════════════════════════════════╝

    Dataset
    ───────
      Training rows  : {n_train}  ({n_patients_train} patients)
      Test rows      : {n_test}  ({n_patients_test} patients)
      Features used  : {n_features}
      Target         : Future CognitiveScore (next visit)

    Overall Regression Metrics (held-out test set)
    ──────────────────────────────────────────────
      Model MAE          : {model_mae:.3f}
      Naive baseline MAE : {naive_mae:.3f}  (predict current score as future score)
      Improvement        : {naive_mae - model_mae:+.3f}
      RMSE               : {rmse:.3f}
      R²                 : {r2:.3f}
      Pearson r          : {pearson_r:.3f}

    Cross-Validation ({5}-fold GroupKFold — no patient leakage)
    ─────────────────────────────────────────────────────────
      MAE   : {cv['MAE_mean']:.3f} ± {cv['MAE_std']:.3f}
      RMSE  : {cv['RMSE_mean']:.3f} ± {cv['RMSE_std']:.3f}
      R²    : {cv['R2_mean']:.3f} ± {cv['R2_std']:.3f}

    Progression Label Agreement
    ───────────────────────────
      Overall agreement  : {agreement:.1%}
      (Threshold ±{utils.PROGRESSION_THRESHOLD} points for Improving / Deteriorating)

    Confusion Matrix (rows = actual, columns = predicted)
    ─────────────────────────────────────────────────────
    {"".join("  "+l+chr(10) for l in cm_lines)}

    Subgroup MAE — By Disease
    ─────────────────────────
    {disease_lines}

    Subgroup MAE — By Actual Progression Label
    ──────────────────────────────────────────
    {label_lines}

    Top 15 Most Important Features (mean |SHAP|, test set)
    ───────────────────────────────────────────────────────
    {top_feat_lines}

    Outputs
    ───────
      predictions_test.csv       — test-set predictions and labels
      predictions_all.csv        — one row per patient (used by chatbot)
      feature_importance.csv     — all 82 features ranked by mean |SHAP|
      shap_beeswarm.png          — SHAP beeswarm (top 15)
      shap_bar.png               — SHAP bar chart (top 15)
      shap_waterfall_sample.png  — SHAP waterfall for one patient
      shap_dependence.png        — SHAP dependence plots (top 3 actionable features)
      residual_analysis.png      — predicted vs actual + residual distribution
      confusion_matrix.png       — labelled confusion matrix heatmap
      shap_values.pkl            — SHAP data for phenotype script
      xgb_model.ubj              — saved model weights
    """)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Train the XGBoost model, evaluate it, compute SHAP values, and write all
    outputs to the outputs/ directory.

    Encapsulates all side-effecting work so the module can be imported safely
    without triggering training.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load processed data ───────────────────────────────────────────────────

    print("\nLoading processed data...")
    X_train, X_test, y_train, y_test, test_df, X_all, all_df = run_pipeline(DATA_PATH)
    feature_names = X_train.columns.tolist()

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    # ── Train model ───────────────────────────────────────────────────────────

    print("\nTraining XGBoost model...")

    # Monotone constraints enforce clinical direction for key features so that
    # simulation results are always directionally sensible (e.g. more sleep
    # never lowers the predicted score). Constraint: +1 = higher is better,
    # -1 = higher is worse, 0 = unconstrained.
    _MONOTONE = {
        # Lifestyle — more sleep/steps/mood is better
        "SleepHours": +1, "SleepHours_prev": +1, "SleepHours_delta": +1,
        "SleepHours_roll3": +1, "SleepHours_baseline": +1, "SleepHours_from_baseline": +1,
        "StepsPerDay": +1,
        "MoodScore": +1,
        # Stress — higher is worse
        "StressLevel": -1, "StressLevel_prev": -1, "StressLevel_delta": -1,
        "StressLevel_roll3": -1, "StressLevel_baseline": -1, "StressLevel_from_baseline": -1,
        # Cardiometabolic — elevated values are worse
        "BMI": -1, "BMI_prev": -1, "BMI_delta": -1,
        "BMI_roll3": -1, "BMI_baseline": -1, "BMI_from_baseline": -1,
        "Cholesterol": -1, "Cholesterol_prev": -1, "Cholesterol_delta": -1,
        "Cholesterol_roll3": -1, "Cholesterol_baseline": -1, "Cholesterol_from_baseline": -1,
        "HeartRate": -1, "HeartRate_prev": -1, "HeartRate_delta": -1,
        "HeartRate_roll3": -1, "HeartRate_baseline": -1, "HeartRate_from_baseline": -1,
        "BloodPressure_Systolic": -1, "BloodPressure_Diastolic": -1,
        # Medication adherence — higher is better
        "MedicationAdherence_enc": +1,
        # Smoking — being a smoker is worse
        "Smoker": -1,
    }
    feature_names_list = list(X_train.columns)
    monotone_vec = tuple(_MONOTONE.get(f, 0) for f in feature_names_list)

    model = xgb.XGBRegressor(**XGB_PARAMS, monotone_constraints=monotone_vec)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"Model trained. Best iteration: {model.best_iteration}")

    # ── Core evaluation ───────────────────────────────────────────────────────

    print("\nEvaluating model...")
    preds     = model.predict(X_test)
    model_mae = mean_absolute_error(y_test, preds)
    naive_mae = mean_absolute_error(y_test, test_df["CognitiveScore"])
    rmse      = np.sqrt(mean_squared_error(y_test, preds))
    r2        = r2_score(y_test, preds)
    pearson_r, _ = pearsonr(y_test.values, preds)

    print(f"\n{'='*50}")
    print("TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"Model MAE          : {model_mae:.3f}")
    print(f"Naive baseline MAE : {naive_mae:.3f}")
    print(f"Improvement        : {naive_mae - model_mae:+.3f}")
    print(f"RMSE               : {rmse:.3f}")
    print(f"R²                 : {r2:.3f}")
    print(f"Pearson r          : {pearson_r:.3f}")
    print(f"{'='*50}")

    # ── Cross-validation ──────────────────────────────────────────────────────

    print("\nRunning 5-fold GroupKFold cross-validation...")
    # CV runs on the full preprocessed dataset (no target NaNs) to use all data
    full_df = all_df.dropna(subset=["target_score"])
    X_cv    = full_df[feature_names].fillna(0)
    y_cv    = full_df["target_score"]
    grp_cv  = full_df["PatientID"]
    cv      = cross_validate(X_cv, y_cv, grp_cv, n_splits=5)
    print(f"  CV MAE : {cv['MAE_mean']:.3f} ± {cv['MAE_std']:.3f}")
    print(f"  CV RMSE: {cv['RMSE_mean']:.3f} ± {cv['RMSE_std']:.3f}")
    print(f"  CV R²  : {cv['R2_mean']:.3f} ± {cv['R2_std']:.3f}")

    # ── Progression labels ────────────────────────────────────────────────────

    print("\nGenerating progression labels...")
    results = test_df[["PatientID", "CognitiveScore"]].copy()
    results["actual_future_score"]    = y_test.values
    results["predicted_future_score"] = preds
    results["actual_change"]          = results["actual_future_score"]    - results["CognitiveScore"]
    results["predicted_change"]       = results["predicted_future_score"] - results["CognitiveScore"]
    results["actual_label"]           = results["actual_change"].apply(get_label)
    results["predicted_label"]        = results["predicted_change"].apply(get_label)
    results["disease"]                = test_df.apply(_disease_from_row, axis=1).values

    agreement = (results["actual_label"] == results["predicted_label"]).mean()
    print(f"Category agreement: {agreement:.1%}")

    # Per-disease and per-label subgroup breakdowns
    disease_metrics = _subgroup_metrics(results, "disease")
    label_metrics   = _subgroup_metrics(results, "actual_label")

    print("\nSubgroup MAE by disease:")
    print(disease_metrics.to_string(index=False))
    print("\nSubgroup MAE by progression label:")
    print(label_metrics.to_string(index=False))

    # Labelled confusion matrix (console)
    cm = confusion_matrix(results["actual_label"], results["predicted_label"],
                          labels=LABEL_ORDER)
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"  {'':>15}  " + "  ".join(f"{l:>13}" for l in LABEL_ORDER))
    for i, row_lbl in enumerate(LABEL_ORDER):
        print(f"  Actual {row_lbl:<10}  " + "  ".join(f"{cm[i,j]:>13d}" for j in range(3)))

    # ── SHAP explainability ───────────────────────────────────────────────────

    print("\nGenerating SHAP explanations...")
    explainer = shap.TreeExplainer(model)

    shap_values_test = explainer.shap_values(X_test, check_additivity=False)
    shap_values_all  = explainer.shap_values(X_all,  check_additivity=False)

    importance = pd.Series(
        np.abs(shap_values_test).mean(axis=0),
        index=feature_names,
    ).sort_values(ascending=False)

    print("\nTop 15 Most Important Features (mean |SHAP|):\n")
    print(importance.head(15).round(4).to_string())

    # ── Clean display names for SHAP plots ───────────────────────────────────
    _ROOT_DISPLAY = {
        "CognitiveScore":          "Cognitive Score",
        "BiomarkerScore":          "Biomarker Score",
        "MedicationDose":          "Medication Dose",
        "MedicationAdherence_enc": "Medication Adherence",
        "HeartRate":               "Heart Rate",
        "BloodPressure_Systolic":  "Systolic BP",
        "BloodPressure_Diastolic": "Diastolic BP",
        "Cholesterol":             "Cholesterol",
        "BMI":                     "BMI",
        "SleepHours":              "Sleep Hours",
        "StepsPerDay":             "Steps / Day",
        "StressLevel":             "Stress Level",
        "MoodScore":               "Mood Score",
        "Smoker":                  "Smoker",
        "SupportSystem":           "Support System",
        "HasCaregiver":            "Has Caregiver",
        "Age":                     "Age",
        "Stage":                   "Disease Stage",
        "gap_days":                "Visit Gap (days)",
        "log_gap_days":            "Visit Gap (log)",
        "is_long_gap":             "Long Visit Gap",
        "visit_index":             "Visit Index",
        "days_since_first":        "Days Since First Visit",
    }
    _PLOT_SUFFIXES = ["_from_baseline", "_baseline", "_delta", "_prev", "_roll3"]

    def _plot_name(feat: str) -> str:
        for s in _PLOT_SUFFIXES:
            if feat.endswith(s):
                feat = feat[: -len(s)]
                break
        for prefix in ("Disease_", "Gender_", "MedicalHistory_",
                       "Lifestyle_", "AlcoholUse_", "EmploymentStatus_"):
            if feat.startswith(prefix):
                return feat[len(prefix):].replace("_", " ")
        return _ROOT_DISPLAY.get(feat, feat.replace("_", " "))

    # ── Deduplicate: per unique display name keep the variant with highest mean|SHAP| ──
    name_to_best: dict[str, tuple[int, float]] = {}
    for i, feat in enumerate(feature_names):
        name = _plot_name(feat)
        mean_abs = float(np.abs(shap_values_test[:, i]).mean())
        if name not in name_to_best or mean_abs > name_to_best[name][1]:
            name_to_best[name] = (i, mean_abs)

    sorted_items  = sorted(name_to_best.items(), key=lambda x: x[1][1], reverse=True)[:15]
    top_names     = [item[0] for item in sorted_items]
    top_indices   = [item[1][0] for item in sorted_items]
    filtered_shap = shap_values_test[:, top_indices]
    filtered_x    = X_test.iloc[:, top_indices].copy()
    filtered_x.columns = top_names

    # ── SHAP visualisations ───────────────────────────────────────────────────

    # Beeswarm: importance + direction per patient (deduplicated features)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(filtered_shap, filtered_x, max_display=15, show=False)
    plt.tight_layout()
    beeswarm_path = os.path.join(OUTPUT_DIR, "shap_beeswarm.png")
    plt.savefig(beeswarm_path, dpi=150)
    plt.close()
    print(f"\nSaved: {beeswarm_path}")

    # Bar: global mean absolute SHAP (deduplicated features)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(filtered_shap, filtered_x, plot_type="bar", max_display=15, show=False)
    plt.tight_layout()
    bar_path = os.path.join(OUTPUT_DIR, "shap_bar.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f"Saved: {bar_path}")

    # Dependence plots: show how the top 3 actionable features individually
    # influence predictions, coloured by the feature's most interacting partner.
    # We skip pure cognitive-history features since they are non-actionable.
    _non_actionable_prefixes = ("days_since_first", "visit_index")
    actionable_top = [
        f for f in importance.index
        if not any(f.startswith(p) for p in _non_actionable_prefixes)
    ][:3]

    dep_fig, dep_axes = plt.subplots(1, len(actionable_top), figsize=(6 * len(actionable_top), 5))
    if len(actionable_top) == 1:
        dep_axes = [dep_axes]
    for ax, feat in zip(dep_axes, actionable_top):
        shap.dependence_plot(
            feat, shap_values_test, X_test,
            feature_names=feature_names,
            ax=ax, show=False,
        )
        ax.set_title(f"SHAP dependence: {feat.replace('_', ' ')}", fontsize=10)
    dep_fig.suptitle("SHAP Dependence Plots — Top 3 Actionable Features", fontweight="bold")
    dep_fig.tight_layout()
    dep_path = os.path.join(OUTPUT_DIR, "shap_dependence.png")
    dep_fig.savefig(dep_path, dpi=150)
    plt.close(dep_fig)
    print(f"Saved: {dep_path}")

    # Waterfall: single representative patient (closest to median predicted score)
    median_pred = float(np.median(preds))
    sample_idx  = int(np.argmin(np.abs(preds - median_pred)))
    waterfall_path = os.path.join(OUTPUT_DIR, "shap_waterfall_sample.png")
    _save_waterfall(
        explainer,
        X_test.iloc[[sample_idx]],
        pred_score   = float(preds[sample_idx]),
        actual_score = float(y_test.iloc[sample_idx]),
        path         = waterfall_path,
    )
    print(f"Saved: {waterfall_path}")

    # Residual analysis
    residual_path = os.path.join(OUTPUT_DIR, "residual_analysis.png")
    _save_residual_plot(y_test.values, preds, residual_path)
    print(f"Saved: {residual_path}")

    # Confusion matrix heatmap
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    _save_confusion_matrix(results["actual_label"], results["predicted_label"], cm_path)
    print(f"Saved: {cm_path}")

    # ── Example patient console output ────────────────────────────────────────

    print("\nExample patient explanation (representative patient):\n")
    sample_shap = shap_values_test[sample_idx]
    top_features = (
        pd.Series(sample_shap, index=feature_names)
        .abs()
        .sort_values(ascending=False)
        .head(5)
    )
    print(f"  Predicted : {preds[sample_idx]:.1f}")
    print(f"  Actual    : {y_test.iloc[sample_idx]:.1f}")
    print("  Top drivers:")
    for feat in top_features.index:
        sv  = sample_shap[feature_names.index(feat)]
        dir = "increased" if sv > 0 else "decreased"
        print(f"    • {feat} → {dir} prediction (SHAP={abs(sv):.3f})")

    # ── Predictions for ALL patients ──────────────────────────────────────────

    print("\nGenerating predictions for all patients...")
    preds_all = model.predict(X_all)
    results_all = all_df[["PatientID", "CognitiveScore"]].copy()
    results_all["predicted_future_score"] = preds_all
    results_all["predicted_change"]       = (
        results_all["predicted_future_score"] - results_all["CognitiveScore"]
    )
    results_all["predicted_label"] = results_all["predicted_change"].apply(get_label)

    # One row per patient — most recent visit mirrors chatbot retrieval
    results_all_dedup = (
        results_all
        .groupby("PatientID")
        .last()
        .reset_index()
    )[["PatientID", "predicted_future_score", "predicted_change", "predicted_label"]]

    results_all_path = os.path.join(OUTPUT_DIR, "predictions_all.csv")
    results_all_dedup.to_csv(results_all_path, index=False)
    print(f"Saved: {results_all_path} ({len(results_all_dedup)} patients)")

    # ── Save test-set results ─────────────────────────────────────────────────

    results_path = os.path.join(OUTPUT_DIR, "predictions_test.csv")
    results.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    # ── Save feature importance CSV ───────────────────────────────────────────

    importance_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    importance.reset_index().rename(
        columns={"index": "feature", 0: "mean_abs_shap"}
    ).to_csv(importance_path, index=False)
    print(f"Saved: {importance_path}")

    # ── Save model ────────────────────────────────────────────────────────────

    model_path = os.path.join(OUTPUT_DIR, "xgb_model.ubj")
    model.save_model(model_path)
    print(f"Saved model: {model_path}")

    # ── Save SHAP values for phenotype generation ─────────────────────────────

    shap_path = os.path.join(OUTPUT_DIR, "shap_values.pkl")
    with open(shap_path, "wb") as f:
        pickle.dump({
            "shap_values":   shap_values_all,
            "feature_names": feature_names,
            "X_all":         X_all,
            "all_df":        all_df,
        }, f)
    print(f"Saved SHAP values: {shap_path}")

    # ── Write performance report ──────────────────────────────────────────────

    # Count training patients: all patients with target data not in the test set
    test_pids = set(test_df["PatientID"])
    n_patients_train = int(
        all_df.loc[
            all_df["target_score"].notna() & ~all_df["PatientID"].isin(test_pids),
            "PatientID",
        ].nunique()
    )

    report = _build_report(
        model_mae        = model_mae,
        naive_mae        = naive_mae,
        rmse             = rmse,
        r2               = r2,
        agreement        = agreement,
        pearson_r        = pearson_r,
        cv               = cv,
        results          = results,
        disease_metrics  = disease_metrics,
        label_metrics    = label_metrics,
        importance_top   = importance.head(15),
        n_train          = len(X_train),
        n_test           = len(X_test),
        n_patients_train = n_patients_train,
        n_patients_test  = results["PatientID"].nunique(),
        n_features       = len(feature_names),
    )

    report_path = os.path.join(OUTPUT_DIR, "model_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report: {report_path}")

    # ── Final summary ─────────────────────────────────────────────────────────

    print("\n" + "=" * 55)
    print("FINAL SUMMARY")
    print("=" * 55)
    print(f"Model MAE (test)   : {model_mae:.3f}")
    print(f"Naive baseline MAE : {naive_mae:.3f}")
    print(f"Improvement        : {naive_mae - model_mae:+.3f}")
    print(f"RMSE               : {rmse:.3f}")
    print(f"R²                 : {r2:.3f}")
    print(f"CV MAE             : {cv['MAE_mean']:.3f} ± {cv['MAE_std']:.3f}")
    print(f"Agreement Rate     : {agreement:.1%}")
    print("\nLabel distribution (test set):")
    for lbl in LABEL_ORDER:
        act  = (results["actual_label"]    == lbl).sum()
        pred = (results["predicted_label"] == lbl).sum()
        print(f"  {lbl:<14} actual={act:3d}  predicted={pred:3d}")
    print("=" * 55)


if __name__ == "__main__":
    main()
