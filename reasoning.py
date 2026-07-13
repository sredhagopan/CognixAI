"""
Clinical Reasoning Pipeline — peer comparison and clinical language generation.

This module has three clearly separated sections:

  Section 1 — Feature Metadata
      ACTIONABLE set, _DISPLAY name map, NEVER_RECOMMEND set.
      These are pure data: no computation.

  Section 2 — Statistical Peer Comparison  (was patient_reasoning.py)
      compute_reasoning() compares a patient's SHAP-ranked features against
      patients in the same phenotype, computing percentiles, z-scores, and
      binary prevalence to give the LLM concrete numbers to reason about.

  Section 3 — Clinical Language Generation  (was clinical_reasoning.py)
      generate_clinical_reasoning() converts the structured dict produced by
      compute_reasoning() into plain-English sentences (concerns, protective
      factors, priorities, peer summary, confidence statement).

Both sections are consumed by prompt_builder.py, which assembles them into
the full LLM system prompt.
"""

import numpy as np
import pandas as pd


# =============================================================================
# SECTION 1 — Feature Metadata
# =============================================================================

# Features a patient can realistically change through behaviour or treatment.
# Everything NOT in this set is non-actionable (demographics, disease history,
# time variables).
ACTIONABLE = {
    # Cardiometabolic — root and temporal variants
    "BMI", "BMI_prev", "BMI_delta", "BMI_roll3", "BMI_from_baseline",
    "Cholesterol", "Cholesterol_prev", "Cholesterol_delta",
    "Cholesterol_roll3", "Cholesterol_from_baseline",
    "HeartRate", "HeartRate_prev", "HeartRate_delta",
    "HeartRate_roll3", "HeartRate_from_baseline",
    "BloodPressure_Systolic", "BloodPressure_Diastolic",
    # Biomarkers / labs
    "BiomarkerScore", "BiomarkerScore_prev", "BiomarkerScore_delta",
    "BiomarkerScore_roll3", "BiomarkerScore_from_baseline",
    # Medication
    "MedicationDose", "MedicationDose_prev", "MedicationDose_delta",
    "MedicationDose_roll3", "MedicationDose_from_baseline",
    "MedicationAdherence_enc",
    # Lifestyle / behaviour
    "SleepHours", "SleepHours_prev", "SleepHours_delta",
    "SleepHours_roll3", "SleepHours_from_baseline",
    "StepsPerDay",
    "StressLevel", "StressLevel_prev", "StressLevel_delta",
    "StressLevel_roll3", "StressLevel_from_baseline",
    "MoodScore",
    "Smoker",
    "Lifestyle_Active", "Lifestyle_Sedentary", "Lifestyle_Smoker",
    "Lifestyle_Non-Smoker", "Lifestyle_Occasional Drinker",
    "AlcoholUse_High", "AlcoholUse_Moderate",
    # Visit regularity
    "gap_days", "log_gap_days", "is_long_gap",
}

# Human-readable display names for every model feature.
# Used by both peer comparison and the LLM prompt builder.
_DISPLAY = {
    "Age":                          "Age",
    # Biomarker — all temporal variants displayed as root name
    "BiomarkerScore":               "Biomarker score",
    "BiomarkerScore_prev":          "Biomarker score",
    "BiomarkerScore_delta":         "Biomarker score",
    "BiomarkerScore_roll3":         "Biomarker score",
    "BiomarkerScore_baseline":      "Biomarker score",
    "BiomarkerScore_from_baseline": "Biomarker score",
    # Medication dose
    "MedicationDose":               "Medication dose",
    "MedicationDose_prev":          "Medication dose",
    "MedicationDose_delta":         "Medication dose",
    "MedicationDose_roll3":         "Medication dose",
    "MedicationDose_baseline":      "Medication dose",
    "MedicationDose_from_baseline": "Medication dose",
    "MedicationAdherence_enc":      "Medication adherence",
    # Heart rate
    "HeartRate":                    "Heart rate",
    "HeartRate_prev":               "Heart rate",
    "HeartRate_delta":              "Heart rate",
    "HeartRate_roll3":              "Heart rate",
    "HeartRate_baseline":           "Heart rate",
    "HeartRate_from_baseline":      "Heart rate",
    "BloodPressure_Systolic":       "Systolic blood pressure",
    "BloodPressure_Diastolic":      "Diastolic blood pressure",
    # Cholesterol
    "Cholesterol":                  "Cholesterol",
    "Cholesterol_prev":             "Cholesterol",
    "Cholesterol_delta":            "Cholesterol",
    "Cholesterol_roll3":            "Cholesterol",
    "Cholesterol_baseline":         "Cholesterol",
    "Cholesterol_from_baseline":    "Cholesterol",
    # BMI
    "BMI":                          "BMI",
    "BMI_prev":                     "BMI",
    "BMI_delta":                    "BMI",
    "BMI_roll3":                    "BMI",
    "BMI_baseline":                 "BMI",
    "BMI_from_baseline":            "BMI",
    # Sleep
    "SleepHours":                   "Sleep hours",
    "SleepHours_prev":              "Sleep hours",
    "SleepHours_delta":             "Sleep hours",
    "SleepHours_roll3":             "Sleep hours",
    "SleepHours_baseline":          "Sleep hours",
    "SleepHours_from_baseline":     "Sleep hours",
    "StepsPerDay":                  "Daily steps",
    # Stress
    "StressLevel":                  "Stress level",
    "StressLevel_prev":             "Stress level",
    "StressLevel_delta":            "Stress level",
    "StressLevel_roll3":            "Stress level",
    "StressLevel_baseline":         "Stress level",
    "StressLevel_from_baseline":    "Stress level",
    "MoodScore":                    "Mood score",
    "Smoker":                       "Smoker",
    "HasCaregiver":                 "Has caregiver",
    "gap_days":                     "Visit regularity",
    "log_gap_days":                 "Visit regularity",
    "is_long_gap":                  "Long gap between visits",
    "days_since_first":             "Time in care",
    "visit_index":                  "Visit number",
    "Gender_Female":                "Female sex",
    "Gender_Male":                  "Male sex",
    "Disease_Alzheimer's":          "Alzheimer's disease",
    "Disease_Diabetes":             "Diabetes",
    "Disease_Parkinson's":          "Parkinson's disease",
    "Lifestyle_Active":             "Active lifestyle",
    "Lifestyle_Non-Smoker":         "Non-smoker",
    "Lifestyle_Occasional Drinker": "Occasional drinker",
    "Lifestyle_Sedentary":          "Sedentary lifestyle",
    "Lifestyle_Smoker":             "Smoker (lifestyle)",
    "MedicalHistory_Asthma":        "Asthma history",
    "MedicalHistory_Heart Disease": "Heart disease history",
    "MedicalHistory_Hypertension":  "Hypertension history",
    "MedicalHistory_Stroke":        "Stroke history",
    "MedicalHistory_Unknown":       "Unknown medical history",
    "AlcoholUse_High":              "High alcohol use",
    "AlcoholUse_Moderate":          "Moderate alcohol use",
    "AlcoholUse_Unknown":           "Unknown alcohol use",
    "EmploymentStatus_Employed":    "Employed",
    "EmploymentStatus_Retired":     "Retired",
    "EmploymentStatus_Unemployed":  "Unemployed",
}

# Features that must never appear in actionable recommendations regardless of
# whether they appear in ACTIONABLE. These are immutable characteristics that
# a patient cannot change (age, disease type, cognitive history, etc.).
NEVER_RECOMMEND = {
    "Age",
    "CognitiveScore",
    "CognitiveScore_prev",
    "CognitiveScore_delta",
    "CognitiveScore_roll3",
    "CognitiveScore_baseline",
    "CognitiveScore_from_baseline",
    "Gender_Female",
    "Gender_Male",
    "Disease_Alzheimer's",
    "Disease_Diabetes",
    "Disease_Parkinson's",
    "days_since_first",
    "visit_index",
}

# Features that are technically in the model but are confusing/meaningless to
# patients when surfaced directly in clinical reasoning text.  These internal
# tracking variables should be silently dropped from MAJOR CONCERNS,
# PROTECTIVE FACTORS, and NON-ACTIONABLE CONTEXT lists.
_SUPPRESS_FROM_DISPLAY = {
    "days_since_first",   # "Days in study" — patients can't change how long they've been enrolled
    "visit_index",        # visit counter — an internal sequence number
}

# Key features tracked for longitudinal trend analysis.
_KEY_CLINICAL_FEATURES = [
    "SleepHours",
    "StressLevel",
    "BMI",
    "Cholesterol",
    "HeartRate",
    "BloodPressure_Systolic",
    "BiomarkerScore",
    "MedicationAdherence_enc",
]

# Features where a higher value is clinically better.
_DESIRABLE_HIGH = {"SleepHours", "BiomarkerScore", "MedicationAdherence_enc"}

# Features where a lower value is clinically better.
_DESIRABLE_LOW = {
    "StressLevel", "BMI", "Cholesterol",
    "HeartRate", "BloodPressure_Systolic",
}


def _display(feature: str) -> str:
    """Return the human-readable name for a model feature."""
    return _DISPLAY.get(feature, feature.replace("_", " "))


# =============================================================================
# SECTION 2 — Statistical Peer Comparison
# =============================================================================

def _is_binary(peer_values: np.ndarray) -> bool:
    """Return True if the feature only takes values 0 and 1 across peers."""
    valid = peer_values[~np.isnan(peer_values)]
    return len(valid) > 0 and set(np.unique(valid)).issubset({0.0, 1.0})


def _percentile_rank(peer_values: np.ndarray, patient_value: float) -> int:
    """
    Return the patient's percentile rank within the peer group (0–100).
    A rank of 80 means the patient is higher than 80% of peers.
    """
    valid = peer_values[~np.isnan(peer_values)]
    if len(valid) == 0:
        return 50
    return int(round(np.mean(valid <= patient_value) * 100))


def _pct_tier(pct: int) -> str:
    """Return a qualitative tier label for a peer percentile rank."""
    if pct >= 90:
        return "among the highest of similar patients"
    elif pct >= 70:
        return "notably higher than most similar patients"
    elif pct >= 40:
        return "near the average for similar patients"
    elif pct >= 20:
        return "notably lower than most similar patients"
    else:
        return "among the lowest of similar patients"


def _shap_importance_tier(importance: float) -> str:
    """Return a qualitative importance label for an absolute SHAP value."""
    if importance >= 0.3:
        return "one of the strongest model drivers"
    elif importance >= 0.15:
        return "a notable model factor"
    elif importance >= 0.05:
        return "a moderate model factor"
    else:
        return "a minor model factor"


def _stats_continuous(
    display: str,
    patient_value: float,
    peer_values: np.ndarray,
    shap_value: float,
) -> dict:
    """
    Compute descriptive statistics comparing the patient to their peer group
    for a continuous feature.

    Returns a dict with mean, median, std, percentile, z-score, and a
    plain-English interpretation sentence.
    """
    valid = peer_values[~np.isnan(peer_values)]
    mean   = float(np.mean(valid))
    median = float(np.median(valid))
    std    = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
    pct    = _percentile_rank(valid, patient_value)
    z      = round((patient_value - mean) / std, 2) if std > 0 else 0.0

    direction = "increases" if shap_value > 0 else "decreases"
    position  = _pct_tier(pct)

    interpretation = (
        f"Your {display} ({patient_value:.1f}) is {position}. "
        f"This factor {direction} your predicted cognitive score."
    )

    return {
        "patient_value":    round(patient_value, 2),
        "phenotype_mean":   round(mean, 2),
        "phenotype_median": round(median, 2),
        "phenotype_std":    round(std, 2),
        "percentile":       pct,
        "z_score":          z,
        "interpretation":   interpretation,
    }


def _stats_binary(
    display: str,
    patient_value: float,
    peer_values: np.ndarray,
    shap_value: float,
) -> dict:
    """
    Compute prevalence statistics for a binary (0/1) feature in the peer group.

    Returns a dict with the patient's value, the peer prevalence percentage,
    and a plain-English interpretation sentence.
    """
    valid      = peer_values[~np.isnan(peer_values)]
    prevalence = float(np.mean(valid) * 100) if len(valid) > 0 else 0.0
    has_it     = patient_value >= 0.5
    direction  = "increases" if shap_value > 0 else "decreases"

    if has_it:
        interpretation = (
            f"You have '{display}', which applies to {prevalence:.0f}% of similar patients. "
            f"This factor {direction} your predicted cognitive score."
        )
    else:
        interpretation = (
            f"You do not have '{display}' ({prevalence:.0f}% of similar patients do). "
            f"This factor {direction} your predicted cognitive score."
        )

    return {
        "patient_value":             int(has_it),
        "phenotype_prevalence_pct":  round(prevalence, 1),
        "interpretation":            interpretation,
    }


def compute_reasoning(
    patient_id: str,
    visit_row: pd.Series,
    shap_row: np.ndarray,
    feature_names: list,
    patient_data: pd.DataFrame,
    all_df: pd.DataFrame,
    phenotype: str,
    top_n_actionable: int = 4,
    top_n_non_actionable: int = 3,
) -> dict:
    """
    Compare a patient against their phenotype peers and return a structured
    reasoning dict.

    The dict is consumed by generate_clinical_reasoning() below to produce
    plain-English explanations for the LLM.

    Parameters
    ----------
    patient_id          : Patient being explained.
    visit_row           : The all_df row for the patient's prediction visit (last visit).
    shap_row            : SHAP values array aligned to feature_names (same visit).
    feature_names       : Ordered list of model features (82 features).
    patient_data        : Baseline + predictions table containing Phenotype_Simplified.
    all_df              : Full longitudinal dataset (all rows, all patients).
    phenotype           : Patient's phenotype label.
    top_n_actionable    : How many actionable factors to include in the result.
    top_n_non_actionable: How many non-actionable factors to include.

    Returns
    -------
    dict with keys:
        top_actionable_factors     — list of factor dicts (things patient can change)
        top_non_actionable_factors — list of factor dicts (fixed characteristics)
        n_peers                    — number of peer patients found
        overall_summary            — plain-English summary sentence
        cautions                   — list of caveat strings
    """
    # 1. Identify peer patients — same phenotype, excluding the patient themselves
    peer_ids = set(
        patient_data.loc[
            patient_data["Phenotype_Simplified"] == phenotype, "PatientID"
        ]
    )
    peer_ids.discard(patient_id)

    if not peer_ids:
        return {
            "top_actionable_factors":     [],
            "top_non_actionable_factors": [],
            "n_peers":                    0,
            "overall_summary":            "No peer patients found in this phenotype for comparison.",
            "cautions":                   ["Peer comparison unavailable — only one patient in this phenotype."],
        }

    # Use each peer's last visit to mirror how predictions are made
    peer_last = (
        all_df[all_df["PatientID"].isin(peer_ids)]
        .groupby("PatientID", sort=False)
        .last()
        .reset_index(drop=True)
    )
    n_peers = len(peer_last)

    # 2. Rank all features by absolute SHAP importance
    shap_ranked = sorted(
        zip(feature_names, shap_row.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    # 3. Walk the ranked list and build factor entries for each category
    actionable:     list = []
    non_actionable: list = []

    for feature, shap_val in shap_ranked:
        # Stop when both lists are full
        if len(actionable) >= top_n_actionable and len(non_actionable) >= top_n_non_actionable:
            break

        if feature not in peer_last.columns:
            continue

        raw = visit_row.get(feature)
        if raw is None or pd.isna(raw):
            continue

        try:
            patient_value = float(raw)
            peer_values   = peer_last[feature].to_numpy(dtype=float)
        except (ValueError, TypeError):
            continue

        is_act  = feature in ACTIONABLE
        display = _display(feature)

        # Skip if this category is already full
        if is_act and len(actionable) >= top_n_actionable:
            continue
        if not is_act and len(non_actionable) >= top_n_non_actionable:
            continue

        if _is_binary(peer_values):
            stats = _stats_binary(display, patient_value, peer_values, shap_val)
        else:
            stats = _stats_continuous(display, patient_value, peer_values, shap_val)

        entry = {
            "feature":         feature,
            "display_name":    display,
            "shap_importance": round(abs(float(shap_val)), 4),
            "shap_direction":  "increases" if shap_val > 0 else "decreases",
            **stats,
        }

        if is_act:
            actionable.append(entry)
        else:
            non_actionable.append(entry)

    # 4. Build plain-English overall summary
    summary_parts = [
        f"Comparison against {n_peers} patients in the '{phenotype}' phenotype."
    ]
    if actionable:
        top = actionable[0]
        pct_clause = (
            f", which is {_pct_tier(top['percentile'])} for this phenotype"
            if "percentile" in top else ""
        )
        summary_parts.append(
            f"The most influential actionable factor is {top['display_name']}"
            f"{pct_clause} and is {_shap_importance_tier(top['shap_importance'])}."
        )
    if non_actionable:
        top = non_actionable[0]
        summary_parts.append(
            f"The most influential non-actionable factor is {top['display_name']}"
            f" ({_shap_importance_tier(top['shap_importance'])})."
        )

    cautions = [
        f"Peer statistics are based on {n_peers} patients in the same phenotype.",
        "SHAP values reflect the model's weighting for this specific visit — not a clinical diagnosis.",
        "Actionable vs. non-actionable classification is an approximation; discuss all factors with your care team.",
    ]

    return {
        "top_actionable_factors":     actionable,
        "top_non_actionable_factors": non_actionable,
        "n_peers":                    n_peers,
        "overall_summary":            " ".join(summary_parts),
        "cautions":                   cautions,
    }


def compute_feature_trends(
    patient_id: str,
    all_df: pd.DataFrame,
    n_recent: int = 4,
) -> dict:
    """
    Compute the trend direction for key clinical features over the patient's
    most recent visits.

    Parameters
    ----------
    patient_id : Patient ID string.
    all_df     : Full longitudinal DataFrame (all patients, all visits).
    n_recent   : Number of most recent visits to analyse.

    Returns
    -------
    dict mapping feature_name → {
        direction : "improving" | "worsening" | "stable",
        magnitude : "slight" | "moderate" | "notable",
        n_visits  : int,
        recent    : float (most recent value),
        first     : float (oldest value in the window),
    }

    Only features with ≥ 2 valid values are included.  Direction is assessed
    relative to clinical desirability: a falling cholesterol is "improving"
    because _DESIRABLE_LOW includes it; a rising sleep hours is "improving"
    because _DESIRABLE_HIGH includes it.
    """
    patient_rows = (
        all_df[all_df["PatientID"] == patient_id]
        .sort_values("Date")
        .tail(n_recent)
    )

    if len(patient_rows) < 2:
        return {}

    # (low_threshold, high_threshold) for magnitude classification by slope.
    # Units are "value units per visit interval".
    _thresholds: dict[str, tuple[float, float]] = {
        "Cholesterol":            (3.0, 10.0),
        "BMI":                    (0.3,  1.5),
        "SleepHours":             (0.2,  0.8),
        "StressLevel":            (0.3,  1.0),
        "HeartRate":              (1.5,  4.0),
        "BloodPressure_Systolic": (1.5,  4.0),
        "BiomarkerScore":         (0.5,  2.0),
        "MedicationAdherence_enc":(0.1,  0.5),
    }

    results = {}

    for feat in _KEY_CLINICAL_FEATURES:
        if feat not in patient_rows.columns:
            continue

        values = patient_rows[feat].dropna().to_numpy(dtype=float)
        if len(values) < 2:
            continue

        xs = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(xs, values, 1)[0])

        low_t, high_t = _thresholds.get(feat, (0.2, 1.0))

        if abs(slope) < low_t:
            direction = "stable"
            magnitude = "slight"
        else:
            improving = (
                (feat in _DESIRABLE_HIGH and slope > 0)
                or (feat in _DESIRABLE_LOW and slope < 0)
            )
            direction = "improving" if improving else "worsening"
            magnitude = "notable" if abs(slope) >= high_t else "moderate"

        results[feat] = {
            "direction": direction,
            "magnitude": magnitude,
            "n_visits":  int(len(values)),
            "recent":    float(values[-1]),
            "first":     float(values[0]),
        }

    return results


# =============================================================================
# SECTION 3 — Clinical Language Generation
# =============================================================================

# ── Sentence builders ─────────────────────────────────────────────────────────
# Each builder takes a factor dict (produced by compute_reasoning) and returns
# a single plain-English sentence, or "" if the factor doesn't apply.

def _concern_sentence(factor: dict) -> str:
    """
    Return a plain-English concern sentence for a factor that DECREASES the
    predicted cognitive score, or "" if the factor increases it.
    """
    if factor["shap_direction"] != "decreases":
        return ""
    name      = factor["display_name"]
    pct       = factor.get("percentile")
    patient_val = factor.get("patient_value")
    mean      = factor.get("phenotype_mean")
    prev_pct  = factor.get("phenotype_prevalence_pct")

    if pct is not None and patient_val is not None and mean is not None:
        if pct >= 90:
            position = "among the highest of similar patients"
        elif pct >= 75:
            position = "notably elevated compared to similar patients"
        elif pct <= 10:
            position = "among the lowest of similar patients"
        elif pct <= 25:
            position = "notably low compared to similar patients"
        else:
            position = "near the group average"
        return (
            f"Your {name} is {position}, and the model associated this "
            f"with a lower predicted cognitive score."
        )
    elif prev_pct is not None:
        has_it = bool(patient_val and patient_val >= 0.5)
        if has_it:
            return (
                f"You have '{name}', which the model associated with a lower "
                f"predicted cognitive score."
            )
        return (
            f"The absence of '{name}' was associated with a lower predicted "
            f"cognitive score in the model."
        )
    return (
        f"Your {name} was associated with a lower predicted cognitive score "
        f"by the model."
    )


def _protective_sentence(factor: dict) -> str:
    """
    Return a plain-English protective sentence for a factor that INCREASES the
    predicted cognitive score, or "" if the factor decreases it.
    """
    if factor["shap_direction"] != "increases":
        return ""
    name      = factor["display_name"]
    pct       = factor.get("percentile")
    patient_val = factor.get("patient_value")
    mean      = factor.get("phenotype_mean")
    prev_pct  = factor.get("phenotype_prevalence_pct")

    if pct is not None and patient_val is not None and mean is not None:
        if pct >= 90:
            position = "among the highest of similar patients — a strong positive signal"
        elif pct >= 75:
            position = "in the top range compared to similar patients"
        elif pct <= 10:
            position = "among the lowest of similar patients"
        elif pct <= 25:
            position = "relatively low compared to similar patients"
        else:
            position = "near the group average"
        return (
            f"Your {name} is {position}, which the model associated with a "
            f"better predicted outcome."
        )
    elif prev_pct is not None:
        has_it = bool(patient_val and patient_val >= 0.5)
        if has_it:
            return (
                f"Having '{name}' was associated with a better predicted "
                f"outcome in the model."
            )
        return ""
    return (
        f"Your {name} was associated with a better predicted cognitive score "
        f"by the model."
    )


# Features where a higher value is clinically undesirable regardless of what
# the model says.  If such a feature appears as "protective" (increases score),
# we cannot recommend "maintaining it" — the LLM must flag the clinical nuance.
_CLINICALLY_HIGH_IS_BAD = {
    "Cholesterol", "Cholesterol_prev", "Cholesterol_delta",
    "Cholesterol_roll3", "Cholesterol_from_baseline",
    "BMI", "BMI_prev", "BMI_delta", "BMI_roll3", "BMI_from_baseline",
    "BloodPressure_Systolic", "BloodPressure_Diastolic",
    "HeartRate", "HeartRate_prev",
    "StressLevel", "StressLevel_prev",
    "AlcoholUse_High",
}


def _priority_sentence(factor: dict, rank: int) -> str:
    """
    Return a ranked priority recommendation sentence for an actionable factor
    that is safe to recommend (not in NEVER_RECOMMEND).
    """
    name      = factor["display_name"]
    feature   = factor.get("feature", "")
    pct       = factor.get("percentile")
    shap      = factor["shap_importance"]
    direction = factor["shap_direction"]

    if direction == "decreases":
        effect = "addressing this could support a better cognitive trajectory"
    elif feature in _CLINICALLY_HIGH_IS_BAD:
        # Model says higher is protective, but clinically high is bad.
        # Flag the nuance rather than endorsing high values.
        effect = (
            "the model associated this with a better prediction in this case, "
            "though managing this factor carefully remains clinically important — "
            "discuss the best target with your care team"
        )
    else:
        effect = "this is already beneficial — maintaining it matters"

    if pct is not None:
        if pct >= 90:
            standing = "among the highest of similar patients"
        elif pct >= 75:
            standing = "notably higher than most similar patients"
        elif pct <= 10:
            standing = "among the lowest of similar patients"
        elif pct <= 25:
            standing = "lower than most similar patients"
        else:
            standing = "near the group average"
        standing_clause = f" (currently {standing})"
    else:
        standing_clause = ""

    if shap >= 0.3:
        weight_phrase = "one of the strongest drivers in the model"
    elif shap >= 0.15:
        weight_phrase = "a meaningful factor in the model"
    else:
        weight_phrase = "a contributing factor in the model"

    labels = {
        1: "Top priority",
        2: "Second priority",
        3: "Third priority",
        4: "Also worth discussing",
    }
    label = labels.get(rank, "Also worth discussing")
    return (
        f"{label}: {name}{standing_clause} — {effect}. "
        f"This is {weight_phrase}."
    )


def generate_clinical_reasoning(ctx: dict) -> dict:
    """
    Convert the structured reasoning dict (from compute_reasoning) into
    plain-English clinical sentences grouped by category.

    The LLM uses this output as its primary reference when forming answers.
    It is designed to be pre-synthesised so the LLM synthesises rather than
    recites raw numbers.

    Parameters
    ----------
    ctx : dict returned by prompt_builder.load_patient_context()

    Returns
    -------
    dict with keys:
        major_concerns                  — list of concern sentences
        protective_factors              — list of protective sentences
        actionable_priorities           — list of ranked recommendation sentences
        non_actionable_factors          — list of context sentences for fixed characteristics
        peer_summary                    — one paragraph comparing patient to peers
        confidence_statement            — one sentence on prediction reliability
        recommended_follow_up_questions — list of natural follow-up prompts
    """
    reasoning        = ctx.get("reasoning", {})
    act_factors      = reasoning.get("top_actionable_factors", [])
    non_act_factors  = reasoning.get("top_non_actionable_factors", [])
    n_peers          = reasoning.get("n_peers", 0)
    shap_factors     = ctx.get("shap_factors", [])
    phenotype        = ctx.get("phenotype", "Unknown")
    predicted_label  = ctx.get("predicted_label", "Unknown")
    recommendation   = ctx.get("recommendation", "")

    # Filter out internal tracking variables that are confusing/meaningless to patients
    all_factors = [
        f for f in act_factors + non_act_factors
        if f["feature"] not in _SUPPRESS_FROM_DISPLAY
    ]

    # ── Major concerns ────────────────────────────────────────────────────────
    major_concerns: list[str] = []
    for f in all_factors:
        sent = _concern_sentence(f)
        if sent:
            major_concerns.append(sent)

    # Supplement with SHAP factors not already covered by the reasoning dict
    covered_names = {f["display_name"] for f in all_factors}
    for f in shap_factors:
        if (f["shap_value"] < -0.1
                and f["display_name"] not in covered_names
                and f["feature"] not in _SUPPRESS_FROM_DISPLAY):
            major_concerns.append(
                f"The model also weighted {f['display_name']} as a notable concern "
                f"for this prediction."
            )
            covered_names.add(f["display_name"])

    if not major_concerns:
        if predicted_label == "Improving":
            major_concerns.append(
                "No strong individual concern factors dominate — the model's "
                "positive outlook appears driven by several small favourable signals."
            )
        else:
            major_concerns.append(
                "The model's primary concerns are largely fixed characteristics "
                "(disease history, age) that provide context but cannot be changed."
            )

    # ── Protective factors ────────────────────────────────────────────────────
    protective_factors: list[str] = []
    for f in all_factors:
        sent = _protective_sentence(f)
        if sent:
            protective_factors.append(sent)

    for f in shap_factors:
        if (f["shap_value"] > 0.1
                and f["display_name"] not in covered_names
                and f["feature"] not in _SUPPRESS_FROM_DISPLAY):
            protective_factors.append(
                f"The model also identified {f['display_name']} as contributing "
                f"positively to this prediction."
            )
            covered_names.add(f["display_name"])

    if not protective_factors:
        protective_factors.append(
            "No clearly protective factors were identified among the top model drivers."
        )

    # ── Actionable priorities ─────────────────────────────────────────────────
    actionable_priorities: list[str] = []
    rank = 1
    for f in act_factors:
        if f["feature"] not in NEVER_RECOMMEND:
            actionable_priorities.append(_priority_sentence(f, rank))
            rank += 1

    if not actionable_priorities:
        fallback = (
            f"The top model factors are mostly fixed characteristics. "
            f"The general recommendation for the {phenotype} group is: {recommendation}."
        ) if recommendation else (
            "No clearly actionable factors were identified among the top model drivers."
        )
        actionable_priorities.append(fallback)

    # ── Non-actionable factors (fixed characteristics — context only) ─────────
    non_actionable_factors: list[str] = []
    for f in non_act_factors:
        if f["feature"] in _SUPPRESS_FROM_DISPLAY:
            continue
        name      = f["display_name"]
        direction = f["shap_direction"]
        effect    = "a positive" if direction == "increases" else "a negative"
        non_actionable_factors.append(
            f"{name} had {effect} influence on the prediction — this is a fixed "
            f"characteristic that cannot be changed but provides important context."
        )

    if not non_actionable_factors:
        non_actionable_factors.append(
            "Fixed characteristics (age, disease type, medical history) were not "
            "among the top model drivers for this patient."
        )

    # ── Peer summary ──────────────────────────────────────────────────────────
    peer_parts = [
        f"This patient was compared against {n_peers} patients in the "
        f"'{phenotype}' phenotype group."
    ]
    for f in act_factors[:2]:
        if f["feature"] in _SUPPRESS_FROM_DISPLAY:
            continue
        pct  = f.get("percentile")
        name = f["display_name"]
        if pct is not None:
            if pct >= 90:
                peer_parts.append(f"{name} is among the highest of peers.")
            elif pct >= 80:
                peer_parts.append(f"{name} is notably higher than most peers.")
            elif pct <= 10:
                peer_parts.append(f"{name} is among the lowest of peers.")
            elif pct <= 20:
                peer_parts.append(f"{name} is notably lower than most peers.")
            else:
                peer_parts.append(f"{name} is near the peer group average.")
    peer_summary = " ".join(peer_parts)

    # ── Confidence statement ──────────────────────────────────────────────────
    # Three dimensions: peer count, predicted-change magnitude vs CV MAE,
    # and SHAP concentration (one dominant factor = clearer picture).
    max_shap = max((f["shap_importance"] for f in all_factors), default=0.0)
    predicted_change = ctx.get("predicted_change")
    _CV_MAE = 4.45  # 5-fold GroupKFold cross-validation result

    if n_peers < 20:
        confidence_level = "Low"
        confidence_statement = (
            f"Confidence is LOW — only {n_peers} similar patients are available "
            f"for comparison, making statistical estimates less reliable."
        )
    elif predicted_change is not None and abs(float(predicted_change)) < _CV_MAE * 0.5:
        # Predicted change is small relative to model error: label direction uncertain
        confidence_level = "Moderate"
        confidence_statement = (
            "Confidence is MODERATE. The predicted change is small relative to the model's "
            "typical margin of error, so the exact direction of change is less certain. "
            "This patient is near-stable — interpret trajectory with appropriate caution."
        )
    elif max_shap >= 0.2 and n_peers >= 50:
        confidence_level = "High"
        confidence_statement = (
            f"Confidence is MODERATE-TO-HIGH. A clear primary driver was identified "
            f"with {n_peers} peers supporting the comparison. "
            f"All predictions carry inherent uncertainty — "
            f"review findings with a clinician before acting."
        )
    else:
        confidence_level = "Moderate"
        confidence_statement = (
            f"Confidence is MODERATE. The prediction reflects available clinical data "
            f"from {n_peers} similar patients. "
            f"Multiple factors contribute rather than one dominant driver — "
            f"interpret with appropriate caution."
        )

    # ── Recommended follow-up questions ───────────────────────────────────────
    follow_ups: list[str] = []

    if act_factors:
        top_concern = act_factors[0]["display_name"]
        follow_ups.append(
            f"Would you like to know more about why {top_concern} matters "
            f"for cognitive health?"
        )

    if n_peers > 0:
        follow_ups.append(
            "Would you like to see how your values compare in detail against "
            "similar patients?"
        )

    if actionable_priorities:
        follow_ups.append(
            "Would you like a more detailed explanation of which factors you "
            "can realistically improve?"
        )

    if protective_factors and "No clearly protective" not in protective_factors[0]:
        follow_ups.append(
            "Would you like to understand which factors are currently working "
            "in your favour?"
        )

    follow_ups.append(
        f"Would you like to understand more about what the '{phenotype}' "
        f"phenotype means for patients like you?"
    )

    # Keep at most 4 suggestions
    follow_ups = follow_ups[:4]

    return {
        "major_concerns":                   major_concerns,
        "protective_factors":               protective_factors,
        "actionable_priorities":            actionable_priorities,
        "non_actionable_factors":           non_actionable_factors,
        "peer_summary":                     peer_summary,
        "confidence_level":                 confidence_level,
        "confidence_statement":             confidence_statement,
        "recommended_follow_up_questions":  follow_ups,
    }
