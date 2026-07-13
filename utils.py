"""
Shared constants and utility functions used across multiple modules.
"""

# Minimum absolute change in predicted cognitive score to classify a patient
# as Improving or Deteriorating. Scores between -2 and +2 are labelled Stable.
PROGRESSION_THRESHOLD = 2

# XGBoost hyperparameters shared by xgboost_and_shap.py and simulation_engine.py.
# Update this dict to change model configuration in both places simultaneously.
XGB_PARAMS: dict = {
    "n_estimators":       300,
    "learning_rate":      0.05,
    "max_depth":          4,
    "subsample":          0.8,
    "colsample_bytree":   0.8,
    "early_stopping_rounds": 20,
    "random_state":       42,
    "n_jobs":             -1,
}

# Ordinal encoding map for MedicationAdherence (0=Low, 1=Medium, 2=High).
ADHERENCE_LABELS: dict = {0.0: "Low", 1.0: "Medium", 2.0: "High"}


def get_label(change: float) -> str:
    """
    Convert a predicted cognitive score change into a clinical progression label.

    Parameters
    ----------
    change : float
        predicted_future_score minus current CognitiveScore.
        Positive values mean the model expects improvement.

    Returns
    -------
    str
        One of "Improving", "Stable", or "Deteriorating".
    """
    if change <= -PROGRESSION_THRESHOLD:
        return "Deteriorating"
    if change >= PROGRESSION_THRESHOLD:
        return "Improving"
    return "Stable"
