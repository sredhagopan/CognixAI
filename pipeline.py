"""
Data Preprocessing Pipeline
============================

Transforms the raw chronic-disease CSV into clean, feature-rich DataFrames
ready for XGBoost training. Every step is a small, named function so each
transformation can be read, tested, and modified independently.

Consumed by:
    xgboost_and_shap.py  — calls run_pipeline() for training and full-cohort SHAP
    simulation_engine.py — calls run_pipeline() if no cached model is found

Input:
    chronic_disease_progression.csv

Output (in-memory only):
    X_train, X_test, y_train, y_test, test_df, X_all, all_df
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error


# =============================================================================
# 1. LOAD DATA
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    """
    Read the CSV, parse dates, and sort chronologically within each patient.

    Parameters
    ----------
    path : str
        Path to chronic_disease_progression.csv.

    Returns
    -------
    pd.DataFrame sorted by (PatientID, Date).
    """
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["PatientID", "Date"])
    return df


# =============================================================================
# 2. CLEAN DATA
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data-quality issues in the raw CSV.

    Steps
    -----
    - Replace negative BiomarkerScore values with NaN (physiologically impossible).
    - Forward- then back-fill BiomarkerScore within each patient's timeline.
    - Fill missing MedicalHistory and AlcoholUse with "Unknown" so they are
      treated as a valid category rather than removed.

    Parameters
    ----------
    df : pd.DataFrame — output of load_data().

    Returns
    -------
    Cleaned copy of df.
    """
    df = df.copy()

    df.loc[df["BiomarkerScore"] < 0, "BiomarkerScore"] = np.nan

    df["BiomarkerScore"] = (
        df.groupby("PatientID")["BiomarkerScore"]
        .transform(lambda x: x.ffill().bfill())
    )

    df["MedicalHistory"] = df["MedicalHistory"].fillna("Unknown")
    df["AlcoholUse"]     = df["AlcoholUse"].fillna("Unknown")

    return df


# =============================================================================
# 3. TIME FEATURES
# =============================================================================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer visit-timing features that capture irregularity in follow-up schedules.

    Features added
    --------------
    gap_days        : days since the previous visit (0 for the first visit).
    log_gap_days    : log1p of gap_days — compresses the right-skewed distribution.
    is_long_gap     : 1 if gap_days > 180 (possible hospitalisation or dropout signal).
    days_since_first: calendar days elapsed since the patient's first visit.
    visit_index     : 0-based visit counter per patient.

    Parameters
    ----------
    df : pd.DataFrame — output of clean_data().

    Returns
    -------
    df with five new columns.
    """
    df = df.copy()

    prev_date = df.groupby("PatientID")["Date"].shift(1)

    df["gap_days"]       = (df["Date"] - prev_date).dt.days.fillna(0)
    df["log_gap_days"]   = np.log1p(df["gap_days"])
    df["is_long_gap"]    = (df["gap_days"] > 180).astype(int)

    first_visit          = df.groupby("PatientID")["Date"].transform("first")
    df["days_since_first"] = (df["Date"] - first_visit).dt.days

    df["visit_index"]    = df.groupby("PatientID").cumcount()

    return df


# =============================================================================
# 4. LONGITUDINAL FEATURES
# =============================================================================

def add_longitudinal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each of eight clinically important biomarkers, compute four derived
    features that let the model learn trends as well as absolute values.

    Derived features per column (e.g. for CognitiveScore)
    -------------------------------------------------------
    CognitiveScore_prev          : value at the previous visit.
    CognitiveScore_delta         : change from the previous visit.
    CognitiveScore_roll3         : rolling 3-visit mean.
    CognitiveScore_baseline      : mean of the patient's first two visits.
    CognitiveScore_from_baseline : current value minus that baseline mean.

    Columns processed
    -----------------
    CognitiveScore, BiomarkerScore, MedicationDose, StressLevel, SleepHours,
    BMI (vascular risk proxy), Cholesterol (cardiovascular-cognitive link),
    HeartRate (autonomic nervous system proxy).

    Parameters
    ----------
    df : pd.DataFrame — output of add_time_features().

    Returns
    -------
    df with 5 × 8 = 40 new columns.
    """
    df = df.copy()

    important_cols = [
        # CognitiveScore is included ONLY to compute its two informative
        # longitudinal features. The raw score and short-term derivatives are
        # excluded in get_features(); only _baseline and _from_baseline survive.
        "CognitiveScore",
        "BiomarkerScore",
        "MedicationDose",
        "StressLevel",
        "SleepHours",
        "BMI",
        "Cholesterol",
        "HeartRate",
    ]

    for col in important_cols:
        group = df.groupby("PatientID")[col]

        df[f"{col}_prev"]          = group.shift(1)
        df[f"{col}_delta"]         = group.diff()
        df[f"{col}_roll3"]         = group.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_baseline"]      = group.transform(lambda x: x.iloc[:2].mean())
        df[f"{col}_from_baseline"] = df[col] - df[f"{col}_baseline"]

    return df


# =============================================================================
# 5. CREATE FUTURE TARGET
# =============================================================================

def create_target(df: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """
    Shift CognitiveScore forward by k visits to create the supervised target.

    Rows for a patient's last k visits will have NaN targets and are removed
    during the train/test split.

    Parameters
    ----------
    df : pd.DataFrame — output of add_longitudinal_features().
    k  : int — prediction horizon (default 1 = next visit).

    Returns
    -------
    df with two new columns: target_score and target_delta.
    """
    df = df.copy()

    df["target_score"] = df.groupby("PatientID")["CognitiveScore"].shift(-k)
    df["target_delta"] = df["target_score"] - df["CognitiveScore"]

    return df


# =============================================================================
# 6. ENCODE CATEGORICAL DATA
# =============================================================================

def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables for XGBoost.

    MedicationAdherence — ordinal (Low=0, Medium=1, High=2) because the
    ordering is meaningful and one-hot would lose that information.

    All other categoricals — one-hot encoded. The raw string columns are
    dropped to avoid duplication with their encoded counterparts.

    Parameters
    ----------
    df : pd.DataFrame — output of create_target().

    Returns
    -------
    df with encoded columns added and raw categorical columns removed.
    """
    df = df.copy()

    adherence_order = [["Low", "Medium", "High"]]
    encoder = OrdinalEncoder(categories=adherence_order)
    df["MedicationAdherence_enc"] = encoder.fit_transform(df[["MedicationAdherence"]])

    categorical_cols = [
        "Gender",
        "Disease",
        "Lifestyle",
        "MedicalHistory",
        "AlcoholUse",
        "EmploymentStatus",
    ]
    df = pd.get_dummies(df, columns=categorical_cols)

    return df


# =============================================================================
# 7. TRAIN / TEST SPLIT
# =============================================================================

def patient_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into train and test sets, guaranteeing no patient appears in both.

    Rows without a future target (last visit per patient) are removed first.
    GroupShuffleSplit groups by PatientID so all visits of a given patient land
    exclusively in either train or test.

    Parameters
    ----------
    df : pd.DataFrame — output of encode_data() with target columns present.

    Returns
    -------
    (train_df, test_df) — two DataFrames with no overlapping patients.
    """
    df = df.dropna(subset=["target_score"])

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["PatientID"]))

    train_df = df.iloc[train_idx]
    test_df  = df.iloc[test_idx]

    print(f"  Train: {len(train_df)} rows | {train_df['PatientID'].nunique()} patients")
    print(f"  Test:  {len(test_df)} rows | {test_df['PatientID'].nunique()} patients")

    return train_df, test_df


# =============================================================================
# 8. SELECT FEATURES
# =============================================================================

def get_features(df: pd.DataFrame) -> list[str]:
    """
    Return the list of model feature columns, excluding non-feature columns.

    Excluded columns
    ----------------
    PatientID, Date           — identifiers, not features.
    target_score, target_delta — prediction targets (would leak the answer).
    Stage                      — changes each visit; acts as a label rather
                                  than an independent measurement.
    MedicationAdherence        — raw string excluded; encoded version
                                  MedicationAdherence_enc is used instead.
    SupportSystem              — free-text field, not encoded.

    Parameters
    ----------
    df : pd.DataFrame — any stage-processed DataFrame with all columns present.

    Returns
    -------
    list[str] of column names to pass to X_train, X_test, X_all.
    """
    exclude_cols = {
        "PatientID",
        "Date",
        "target_score",
        "target_delta",
        "Stage",
        "MedicationAdherence",
        "SupportSystem",
        # CognitiveScore and its short-term derivatives are excluded:
        #   - raw score has r=+0.006 with target (pure visit noise)
        #   - _prev / _delta / _roll3 all have r<0.02 (no real signal)
        # Only _baseline (r=+0.17, patient's cognitive reserve) and
        # _from_baseline (r=-0.11, cumulative trajectory) are kept — they are
        # patient-level characteristics, not the target value.
        "CognitiveScore",
        "CognitiveScore_prev",
        "CognitiveScore_delta",
        "CognitiveScore_roll3",
    }
    return [col for col in df.columns if col not in exclude_cols]


# =============================================================================
# 9. RUN EVERYTHING
# =============================================================================

def run_pipeline(path: str) -> tuple:
    """
    Execute all preprocessing steps in order and return train/test splits.

    Parameters
    ----------
    path : str — path to chronic_disease_progression.csv.

    Returns
    -------
    X_train, X_test, y_train, y_test : feature matrices and target vectors for
        the 80/20 patient-level split.
    test_df  : full processed DataFrame for the test patients (all columns).
    X_all    : feature matrix for the complete dataset (used for all-patient SHAP).
    all_df   : full processed DataFrame for all patients (passed to SHAP script).
    """
    df = load_data(path)
    df = clean_data(df)
    df = add_time_features(df)
    df = add_longitudinal_features(df)
    df = create_target(df)
    df = encode_data(df)

    print("Splitting by patient:")
    train_df, test_df = patient_split(df)

    feature_cols = get_features(df)

    # All-patient matrices (for full-cohort SHAP and chatbot predictions)
    X_all  = df[feature_cols].fillna(0)
    all_df = df.copy()

    # NaN fill: only a handful of derived features (e.g. *_prev, *_delta) are NaN
    # on the first visit; these are safely zeroed because the model has visit_index
    # and is_long_gap to contextualise first-visit rows.
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["target_score"]

    X_test  = test_df[feature_cols].fillna(0)
    y_test  = test_df["target_score"]

    naive_mae = mean_absolute_error(y_test, test_df["CognitiveScore"])
    print(f"  Naive baseline MAE: {naive_mae:.3f}  (predict current score)")
    print("Pipeline complete")

    return X_train, X_test, y_train, y_test, test_df, X_all, all_df


# =============================================================================
# STANDALONE VERIFICATION
# =============================================================================

if __name__ == "__main__":

    DATA_PATH = "chronic_disease_progression.csv"

    X_train, X_test, y_train, y_test, test_df, X_all, all_df = run_pipeline(DATA_PATH)

    print("\nPipeline verified. Ready for model training.")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"X_all   : {X_all.shape}")
    print(f"all_df  : {all_df.shape}")
