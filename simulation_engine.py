"""
Simulation Engine — What-If Scenario Analysis
=============================================

Three query modes:

  1. Specific change   — {feature: new_value} dict built from the user's message
  2. Best single       — "__best_single__"     sweep REALISTIC_TARGETS, return the
                          one change that most improves the predicted score
  3. Best combination  — "__best_combination__" greedy top-N changes

Entry points consumed by llm_chatbot.py:
  is_whatif_question(q)                  — detect if the user's message is a what-if query
  parse_whatif_question(q, ctx, shap_data) — parse into a changes spec or sentinel string
  format_simulation_result(sim)          — format the result for LLM injection
"""

import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from pipeline import run_pipeline
from utils import get_label, XGB_PARAMS, ADHERENCE_LABELS
from reasoning import _display as _display_name

OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "xgb_model.ubj")
DATA_PATH  = "chronic_disease_progression.csv"


# =============================================================================
# Feature aliases and sweep targets
# =============================================================================

# Maps natural-language phrases (lower-case) to model feature names.
FEATURE_ALIASES: dict[str, str] = {
    "cholesterol":              "Cholesterol",
    "bmi":                      "BMI",
    "body mass index":          "BMI",
    "weight":                   "BMI",
    "heart rate":               "HeartRate",
    "heartrate":                "HeartRate",
    "heart":                    "HeartRate",
    "sleep":                    "SleepHours",
    "sleep hours":              "SleepHours",
    "hours of sleep":           "SleepHours",
    "steps":                    "StepsPerDay",
    "steps per day":            "StepsPerDay",
    "daily steps":              "StepsPerDay",
    "exercise":                 "StepsPerDay",
    "activity":                 "StepsPerDay",
    "stress":                   "StressLevel",
    "stress level":             "StressLevel",
    "medication dose":          "MedicationDose",
    "medication":               "MedicationDose",
    "dose":                     "MedicationDose",
    "adherence":                "MedicationAdherence_enc",
    "medication adherence":     "MedicationAdherence_enc",
    "biomarker":                "BiomarkerScore",
    "biomarker score":          "BiomarkerScore",
    "mood":                     "MoodScore",
    "mood score":               "MoodScore",
    "blood pressure":           "BloodPressure_Systolic",
    "systolic":                 "BloodPressure_Systolic",
    "systolic blood pressure":  "BloodPressure_Systolic",
    "diastolic":                "BloodPressure_Diastolic",
    "diastolic blood pressure": "BloodPressure_Diastolic",
    "smoker":                   "Smoker",
    "smoking":                  "Smoker",
}

# Candidate values swept when searching for the best improvement.
# Only features that are also in reasoning.ACTIONABLE and not in NEVER_RECOMMEND
# are used during optimisation.
REALISTIC_TARGETS: dict[str, list] = {
    "BMI":                     [18.5, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
    "Cholesterol":             [150.0, 170.0, 190.0, 200.0, 210.0, 220.0, 240.0],
    "HeartRate":               [55, 60, 65, 70, 75, 80, 85, 90],
    "SleepHours":              [5.0, 6.0, 7.0, 7.5, 8.0, 8.5, 9.0],
    "StepsPerDay":             [2000, 4000, 5000, 6000, 8000, 10000, 12000],
    "StressLevel":             [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    "MedicationDose":          [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0],
    "MedicationAdherence_enc": [0.0, 1.0, 2.0],
    "BiomarkerScore":          [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
    "MoodScore":               [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    "BloodPressure_Systolic":  [110, 115, 120, 125, 130, 135, 140, 150],
    "BloodPressure_Diastolic": [65, 70, 75, 80, 85, 90],
    "Smoker":                  [0.0, 1.0],
}


# Related features that are likely to co-vary in real-world scenarios.
# When a simulation changes one of these features, the LLM should note that
# real-world improvement would likely involve the linked features too.
FEATURE_CASCADES: dict[str, list[str]] = {
    "SleepHours":             ["StressLevel", "MoodScore"],
    "StressLevel":            ["SleepHours", "BloodPressure_Systolic", "HeartRate"],
    "StepsPerDay":            ["BMI", "BloodPressure_Systolic", "HeartRate"],
    "BMI":                    ["Cholesterol", "BloodPressure_Systolic", "HeartRate"],
    "Cholesterol":            ["BMI", "MedicationDose"],
    "BloodPressure_Systolic": ["HeartRate", "StressLevel"],
    "HeartRate":              ["StepsPerDay", "StressLevel", "BloodPressure_Systolic"],
    "MedicationAdherence_enc":["BiomarkerScore"],
}

# Clinical reference ranges for key features, used to contextualise simulated values.
# Tuples are (label, lower_bound, upper_bound) — inclusive on lower, exclusive on upper.
CLINICAL_RANGES: dict[str, list[tuple[str, float, float]]] = {
    "Cholesterol": [
        ("optimal",           0,   200),
        ("borderline high", 200,   240),
        ("high",            240, 99999),
    ],
    "BMI": [
        ("underweight",   0,    18.5),
        ("healthy",      18.5,  25.0),
        ("overweight",   25.0,  30.0),
        ("obese",        30.0, 99999),
    ],
    "BloodPressure_Systolic": [
        ("normal",     0,   120),
        ("elevated", 120,   130),
        ("high",     130, 99999),
    ],
    "SleepHours": [
        ("insufficient",  0,   6),
        ("recommended",   6,   9),
        ("excessive",     9, 999),
    ],
    "StressLevel": [
        ("low",      0, 3),
        ("moderate", 3, 6),
        ("high",     6, 10),
    ],
    "StepsPerDay": [
        ("sedentary",      0,     5000),
        ("lightly active", 5000,  7500),
        ("active",         7500, 10000),
        ("very active",   10000, 999999),
    ],
    "HeartRate": [
        ("low",     0,  60),
        ("normal", 60, 100),
        ("high",  100, 999),
    ],
}


def _clinical_range_label(feat: str, value: float) -> str:
    """Return the clinical range label for a feature value, or '' if unknown."""
    for label, lo, hi in CLINICAL_RANGES.get(feat, []):
        if lo <= value < hi:
            return label
    return ""


# =============================================================================
# SimulationEngine
# =============================================================================

class SimulationEngine:
    """
    Wraps the trained XGBoost model for what-if scenario analysis.

    The model is loaded lazily from MODEL_PATH on first access.
    If the file does not exist, the engine re-trains and saves it.
    """

    def __init__(self):
        self._model: xgb.XGBRegressor | None = None
        self._explainer: shap.TreeExplainer | None = None

    # ── Model loading ─────────────────────────────────────────────────────────

    def _ensure_model(self) -> xgb.XGBRegressor:
        if self._model is not None:
            return self._model

        if os.path.exists(MODEL_PATH):
            m = xgb.XGBRegressor()
            m.load_model(MODEL_PATH)
        else:
            print("[SimulationEngine] No cached model found — training now...")
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            X_train, X_test, y_train, y_test, _, _, _ = run_pipeline(DATA_PATH)
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            m.save_model(MODEL_PATH)
            print(f"[SimulationEngine] Model saved to {MODEL_PATH}")

        self._model = m
        self._explainer = shap.TreeExplainer(m)
        return m

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_patient_vector(self, patient_id: str, shap_data: dict) -> pd.Series:
        """Return the last-visit feature vector for the patient as a named Series."""
        all_df = shap_data["all_df"]
        feature_names = shap_data["feature_names"]
        indices = np.where(all_df["PatientID"].values == patient_id)[0]
        if len(indices) == 0:
            raise ValueError(f"Patient {patient_id!r} not found in SHAP data.")
        return all_df.iloc[indices[-1]][feature_names].fillna(0).copy()

    def _current_cognitive_score(self, patient_id: str, shap_data: dict) -> float | None:
        all_df = shap_data["all_df"]
        rows = all_df[all_df["PatientID"] == patient_id]
        if rows.empty:
            return None
        val = rows.iloc[-1].get("CognitiveScore")
        return float(val) if val is not None and not pd.isna(val) else None

    # ── Core simulation ───────────────────────────────────────────────────────

    def simulate(
        self,
        patient_id: str,
        changes: dict,
        shap_data: dict,
    ) -> dict:
        """
        Apply `changes` to the patient's feature vector and re-predict.

        Returns a result dict with before/after scores, delta, label change,
        and the top SHAP shifts caused by the modification.
        """
        model = self._ensure_model()
        feature_names = shap_data["feature_names"]

        base_vec = self._get_patient_vector(patient_id, shap_data)
        current_cog = self._current_cognitive_score(patient_id, shap_data)

        X_before = base_vec.values.reshape(1, -1)
        before_score = float(model.predict(X_before)[0])

        mod_vec = base_vec.copy()
        for feat, val in changes.items():
            if feat in mod_vec.index:
                mod_vec[feat] = float(val)

        # Cascade root feature changes to their temporal derivatives so the
        # model sees a self-consistent feature vector (e.g. changing SleepHours
        # should also update SleepHours_delta and SleepHours_from_baseline).
        for feat, val in changes.items():
            new_val = float(val)
            prev_feat     = f"{feat}_prev"
            delta_feat    = f"{feat}_delta"
            roll3_feat    = f"{feat}_roll3"
            baseline_feat = f"{feat}_baseline"
            fromb_feat    = f"{feat}_from_baseline"

            prev_val     = float(base_vec[prev_feat])     if prev_feat     in mod_vec.index else None
            baseline_val = float(base_vec[baseline_feat]) if baseline_feat in mod_vec.index else None

            if delta_feat in mod_vec.index and prev_val is not None:
                mod_vec[delta_feat] = new_val - prev_val
            if fromb_feat in mod_vec.index and baseline_val is not None:
                mod_vec[fromb_feat] = new_val - baseline_val
            if roll3_feat in mod_vec.index and prev_val is not None:
                mod_vec[roll3_feat] = (new_val + prev_val + prev_val) / 3  # prev as proxy for t-2

        X_after = mod_vec.values.reshape(1, -1)
        after_score = float(model.predict(X_after)[0])
        delta = after_score - before_score

        shap_before = self._explainer.shap_values(X_before, check_additivity=False)[0]
        shap_after  = self._explainer.shap_values(X_after,  check_additivity=False)[0]

        shap_shifts = []
        for i, feat in enumerate(feature_names):
            shift = float(shap_after[i] - shap_before[i])
            if abs(shift) > 1e-4:
                shap_shifts.append({
                    "feature":      feat,
                    "display_name": _display_name(feat),
                    "before":       round(float(shap_before[i]), 4),
                    "after":        round(float(shap_after[i]),  4),
                    "shift":        round(shift, 4),
                })
        shap_shifts.sort(key=lambda x: abs(x["shift"]), reverse=True)

        cog = current_cog or 0.0
        before_label = get_label(before_score - cog)
        after_label  = get_label(after_score  - cog)

        original_values = {
            feat: round(float(base_vec[feat]), 4)
            for feat in changes
            if feat in base_vec.index
        }

        return {
            "patient_id":      patient_id,
            "changes_applied": changes,
            "original_values": original_values,
            "before_score":    round(before_score, 2),
            "after_score":     round(after_score,  2),
            "delta":           round(delta, 2),
            "before_label":    before_label,
            "after_label":     after_label,
            "label_changed":   before_label != after_label,
            "shap_shifts":     shap_shifts[:5],
        }

    # ── Optimisation ──────────────────────────────────────────────────────────

    def find_best_single_change(
        self,
        patient_id: str,
        shap_data: dict,
    ) -> dict:
        """
        Sweep REALISTIC_TARGETS across all actionable features and return the
        single change that most improves the predicted cognitive score.
        """
        from reasoning import ACTIONABLE, NEVER_RECOMMEND

        model = self._ensure_model()
        feature_names = shap_data["feature_names"]
        base_vec = self._get_patient_vector(patient_id, shap_data)
        baseline_pred = float(model.predict(base_vec.values.reshape(1, -1))[0])

        best_delta   = 0.0
        best_feature = None
        best_value   = None

        for feat, candidates in REALISTIC_TARGETS.items():
            if feat not in feature_names:
                continue
            if feat not in ACTIONABLE or feat in NEVER_RECOMMEND:
                continue

            for val in candidates:
                trial = base_vec.copy()
                trial[feat] = float(val)
                pred = float(model.predict(trial.values.reshape(1, -1))[0])
                if pred - baseline_pred > best_delta:
                    best_delta   = pred - baseline_pred
                    best_feature = feat
                    best_value   = val

        if best_feature is None:
            return {
                "mode":          "best_single",
                "found":         False,
                "before_score":  round(baseline_pred, 2),
                "after_score":   round(baseline_pred, 2),
                "delta":         0.0,
                "best_feature":  None,
                "best_value":    None,
                "changes_applied": {},
                "shap_shifts":   [],
                "before_label":  get_label(0.0),
                "after_label":   get_label(0.0),
                "label_changed": False,
            }

        result = self.simulate(patient_id, {best_feature: best_value}, shap_data)
        result["mode"]         = "best_single"
        result["found"]        = True
        result["best_feature"] = best_feature
        result["best_value"]   = best_value
        return result

    def find_best_combination(
        self,
        patient_id: str,
        shap_data: dict,
        top_n: int = 3,
    ) -> dict:
        """
        Greedy search: at each step pick the remaining feature change that most
        improves the cumulative prediction, up to top_n steps.
        """
        from reasoning import ACTIONABLE, NEVER_RECOMMEND

        model = self._ensure_model()
        feature_names = shap_data["feature_names"]
        current_vec   = self._get_patient_vector(patient_id, shap_data).copy()
        baseline_pred = float(model.predict(current_vec.values.reshape(1, -1))[0])
        current_pred  = baseline_pred
        chosen: dict  = {}
        individual_impacts: list = []  # ranked per-feature contribution

        for _ in range(top_n):
            best_delta   = 0.0
            best_feature = None
            best_value   = None

            for feat, candidates in REALISTIC_TARGETS.items():
                if feat not in feature_names:
                    continue
                if feat not in ACTIONABLE or feat in NEVER_RECOMMEND:
                    continue
                if feat in chosen:
                    continue

                for val in candidates:
                    trial = current_vec.copy()
                    trial[feat] = float(val)
                    pred = float(model.predict(trial.values.reshape(1, -1))[0])
                    if pred - current_pred > best_delta:
                        best_delta   = pred - current_pred
                        best_feature = feat
                        best_value   = val

            if best_feature is None:
                break

            current_vec[best_feature] = float(best_value)
            current_pred = float(model.predict(current_vec.values.reshape(1, -1))[0])
            chosen[best_feature] = best_value
            individual_impacts.append({
                "feature":      best_feature,
                "value":        best_value,
                "delta":        round(best_delta, 3),
            })

        if not chosen:
            return {
                "mode":               "best_combination",
                "found":              False,
                "changes_applied":    {},
                "before_score":       round(baseline_pred, 2),
                "after_score":        round(baseline_pred, 2),
                "delta":              0.0,
                "before_label":       get_label(0.0),
                "after_label":        get_label(0.0),
                "label_changed":      False,
                "shap_shifts":        [],
                "individual_impacts": [],
            }

        result = self.simulate(patient_id, chosen, shap_data)
        result["mode"]               = "best_combination"
        result["found"]              = True
        result["individual_impacts"] = individual_impacts
        return result


# =============================================================================
# What-if detection
# =============================================================================

_WHATIF_PATTERNS = [
    r"\bwhat\s*if\b",
    r"\bwhat\s+would\s+happen\s+if\b",
    r"\bif\s+(?:my|i)\b",
    r"\bif\s+i\s+(?:reduce[d]?|increase[d]?|lower(?:ed)?|raise[d]?|improve[d]?|"
    r"change[d]?|stop(?:ped)?|start(?:ed)?|quit|cut|lost?|gain(?:ed)?|took|sleep)\b",
    r"\bsuppose\b",
    r"\bhypothetically\b",
    r"\bscenario\b",
    r"\bbest\s+(?:single\s+)?change\b",
    r"\bbest\s+(?:thing|action|step|improvement|combination|combo|plan)\b",
    r"\boptimal(?:\s+change)?\b",
    r"\bmost\s+impactful\b",
    r"\bwhat\s+(?:should|can|could)\s+i\s+(?:do|change|improve|adjust)\b",
    r"\bwhat\s+(?:changes?|improvements?)\s+(?:would|could|might)\b",
    r"\bwhat\s+(?:one\s+thing|single\s+thing)\b",
]

_WHATIF_RE = re.compile("|".join(_WHATIF_PATTERNS), re.IGNORECASE)


def is_whatif_question(q: str) -> bool:
    """Return True if the question describes a hypothetical scenario or asks for optimisation."""
    return bool(_WHATIF_RE.search(q))


# =============================================================================
# What-if parsing
# =============================================================================

_BEST_COMBO_RE = re.compile(
    r"\bbest\s+(?:combination|combo|set|mix|plan|bundle|multiple|multi|overall)\b"
    r"|\bwhat\s+(?:should|can|could)\s+i\s+(?:do|change|improve|adjust)\b"
    r"|\bwhat\s+(?:changes?|improvements?)\s+(?:would|could|might)\b",
    re.IGNORECASE,
)

_BEST_SINGLE_RE = re.compile(
    r"\bbest\s+(?:single\s+)?(?:change|thing|action|step|improvement)\b"
    r"|\bmost\s+impactful\b"
    r"|\bgreatest\s+(?:single\s+)?(?:change|impact|improvement)\b"
    r"|\boptimal(?:\s+change)?\b"
    r"|\bwhat\s+(?:one|single)\s+thing\b",
    re.IGNORECASE,
)

# Absolute value: "FEATURE (was/were/is/to/at/of) NUMBER"
_ABS_RE = re.compile(
    r"(?P<feature>[a-zA-Z][a-zA-Z ]{2,30}?)\s+"
    r"(?:was|were|is|be|to|at|of|became?)\s+"
    r"(?P<value>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Downward relative: "FEATURE dropped/fell/reduced/decreased by NUMBER"
_DOWN_RE = re.compile(
    r"(?P<feature>[a-zA-Z][a-zA-Z ]{2,30}?)\s+"
    r"(?:reduced?|lowered?|dropped?|decreased?|fell?|cut(?:\s+down)?)\s+"
    r"(?:down\s+)?(?:by\s+)?(?P<value>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Upward relative: "FEATURE increased/raised/improved by NUMBER"
_UP_RE = re.compile(
    r"(?P<feature>[a-zA-Z][a-zA-Z ]{2,30}?)\s+"
    r"(?:increased?|raised?|improved?|went\s+up|boosted?)\s+"
    r"(?:by\s+)?(?P<value>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# Verb-first: "sleep 8 hours", "walk 10000 steps", "take 50 medication"
_VERB_FIRST_RE = re.compile(
    r"(?:sleep|slept|walk(?:ed)?|take|took|do|did|get|got|reach(?:ed)?|hit|maintain)\s+"
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>hours?|steps?|bpm|mg(?:/dl)?|points?)?",
    re.IGNORECASE,
)

_UNIT_FEATURE = {
    "hour": "SleepHours",
    "step": "StepsPerDay",
    "bpm":  "HeartRate",
}


def _resolve_alias(raw: str) -> str | None:
    """Map a natural-language phrase to a model feature name, or None."""
    key = raw.strip().lower()
    if key in FEATURE_ALIASES:
        return FEATURE_ALIASES[key]
    for alias, feat in FEATURE_ALIASES.items():
        if alias in key or key in alias:
            return feat
    return None


def parse_whatif_question(q: str, ctx: dict, shap_data: dict) -> dict | str:
    """
    Parse a what-if question into one of:
        - "__best_combination__"  — optimise across multiple features
        - "__best_single__"       — optimise for the single best change
        - {feature_name: value}   — a specific scenario to simulate

    Falls back to "__best_single__" when the text is ambiguous.
    """
    # Check for combination intent before single, so "best combination" wins
    if _BEST_COMBO_RE.search(q) and not _BEST_SINGLE_RE.search(q):
        return "__best_combination__"
    if _BEST_SINGLE_RE.search(q):
        return "__best_single__"

    feature_names = shap_data["feature_names"]
    all_df = shap_data["all_df"]
    patient_id = ctx.get("patient_id")

    current_row = None
    if patient_id:
        indices = np.where(all_df["PatientID"].values == patient_id)[0]
        if len(indices) > 0:
            current_row = all_df.iloc[indices[-1]]

    changes: dict = {}

    # Absolute values
    for m in _ABS_RE.finditer(q):
        feat = _resolve_alias(m.group("feature"))
        if feat and feat in feature_names:
            changes[feat] = float(m.group("value"))

    # Downward relative
    if current_row is not None:
        for m in _DOWN_RE.finditer(q):
            feat = _resolve_alias(m.group("feature"))
            if feat and feat in feature_names:
                cur = current_row.get(feat)
                if cur is not None and not pd.isna(cur):
                    changes[feat] = float(cur) - float(m.group("value"))

        # Upward relative
        for m in _UP_RE.finditer(q):
            feat = _resolve_alias(m.group("feature"))
            if feat and feat in feature_names:
                cur = current_row.get(feat)
                if cur is not None and not pd.isna(cur):
                    changes[feat] = float(cur) + float(m.group("value"))

    # Verb-first patterns ("sleep 8 hours", "walk 10000 steps")
    for m in _VERB_FIRST_RE.finditer(q):
        unit = (m.group("unit") or "").lower().rstrip("s")
        for key, feat in _UNIT_FEATURE.items():
            if key in unit and feat in feature_names:
                changes[feat] = float(m.group("value"))

    if changes:
        return changes

    # If we detected a what-if intent but couldn't parse specifics, optimise
    return "__best_single__"


# =============================================================================
# Result formatting
# =============================================================================

def _display_feature(feat: str, value: float | None = None) -> str:
    """
    Return a human-readable label for a feature, with value decoded where needed.
    MedicationAdherence_enc encodes 0→Low, 1→Medium, 2→High.
    """
    from reasoning import _DISPLAY as _FEAT_DISPLAY
    label = _FEAT_DISPLAY.get(feat, feat.replace("_", " "))
    if value is not None and feat == "MedicationAdherence_enc":
        value_label = ADHERENCE_LABELS.get(float(value), str(value))
        return f"{label} → {value_label}"
    if value is not None:
        return f"{label} → {value}"
    return label



def format_simulation_result(sim: dict) -> str:
    """
    Format a simulation result as a structured block for injection into the LLM turn.

    Pre-writes a plain-English interpretation sentence so the LLM only needs to
    present and elaborate on already-computed facts — never estimate or predict.
    """
    from reasoning import _DISPLAY as _FEAT_DISPLAY

    if not sim:
        return ""

    mode         = sim.get("mode", "specific")
    found        = sim.get("found", True)
    before       = sim.get("before_score", 0.0)
    after        = sim.get("after_score",  0.0)
    delta        = sim.get("delta", 0.0)
    label_before = sim.get("before_label", "")
    label_after  = sim.get("after_label",  "")
    label_change = sim.get("label_changed", False)
    changes      = sim.get("changes_applied", {})

    mode_label = {
        "best_single":      "Best Single Change",
        "best_combination": "Best Combination of Changes",
    }.get(mode, "Specific Scenario")

    lines = [
        f"[WHAT-IF SIMULATION — {mode_label}]",
        "SOURCE OF TRUTH: Computed by simulation_engine.py.",
        "You are the interpreter. Present only what is computed here — do not estimate or extend.",
        "Rules: cite exact scores and delta; never qualify computed results with 'probably' or 'likely';",
        "never use simulated values as a starting point for further arithmetic.",
        "",
    ]

    if not found:
        lines += [
            "RESULT: No beneficial change found among modifiable features.",
            "Explain: values may already be near optimal, or the key drivers are fixed characteristics.",
            "Remind the patient to discuss options with their care team.",
            "[END SIMULATION]",
        ]
        return "\n".join(lines)

    original_values = sim.get("original_values", {})

    # ── What was simulated ────────────────────────────────────────────────────
    if mode == "best_single" and sim.get("best_feature"):
        feat        = sim["best_feature"]
        val         = sim["best_value"]
        feat_label  = _FEAT_DISPLAY.get(feat, feat.replace("_", " "))
        orig        = original_values.get(feat)
        if feat == "MedicationAdherence_enc":
            new_s  = ADHERENCE_LABELS.get(float(val), str(val))
            orig_s = ADHERENCE_LABELS.get(float(orig), str(orig)) if orig is not None else "?"
            lines.append(f"Optimal change: {feat_label}: {orig_s} → {new_s}")
        else:
            rng      = _clinical_range_label(feat, float(val))
            rng_note = f" [{rng}]" if rng else ""
            if orig is not None:
                orig_rng  = _clinical_range_label(feat, float(orig))
                orig_note = f" [{orig_rng}]" if orig_rng else ""
                lines.append(f"Optimal change: {feat_label}: {orig}{orig_note} → {val}{rng_note}")
            else:
                lines.append(f"Optimal change: set {feat_label} → {val}{rng_note}")
    elif changes:
        lines.append("Change simulated:")
        for feat, val in changes.items():
            feat_label = _FEAT_DISPLAY.get(feat, feat.replace("_", " "))
            orig       = original_values.get(feat)
            if feat == "MedicationAdherence_enc":
                new_s  = ADHERENCE_LABELS.get(float(val), str(val))
                orig_s = ADHERENCE_LABELS.get(float(orig), str(orig)) if orig is not None else "?"
                lines.append(f"  {feat_label}: {orig_s} → {new_s}")
            else:
                rng      = _clinical_range_label(feat, float(val))
                rng_note = f" [{rng}]" if rng else ""
                if orig is not None:
                    orig_rng  = _clinical_range_label(feat, float(orig))
                    orig_note = f" [{orig_rng}]" if orig_rng else ""
                    lines.append(f"  {feat_label}: {orig}{orig_note} → {val}{rng_note}")
                else:
                    lines.append(f"  {feat_label} → {val}{rng_note}")

    # ── Computed results ──────────────────────────────────────────────────────
    lines += [
        "",
        "COMPUTED RESULTS:",
        f"  Original score : {before:.1f}  ({label_before})",
        f"  Simulated score: {after:.1f}  ({label_after})",
        f"  Score change   : {delta:+.1f} points",
    ]
    if label_change:
        lines.append(f"  Trajectory     : {label_before} → {label_after}  (trajectory changed)")
    else:
        lines.append(f"  Trajectory     : Unchanged ({label_after})")

    # ── Pre-written plain-English interpretation ──────────────────────────────
    if abs(delta) < 0.5:
        lines += [
            "",
            f"MINIMAL EFFECT: Score changed by only {delta:+.1f} points.",
            "  Tell the patient: this feature is not currently a dominant model driver.",
            "  Do NOT hedge with 'this might still help clinically' — that goes beyond the simulation.",
            "  Clarify: this reflects the model only, not the feature's real-world clinical importance.",
        ]
    else:
        magnitude = (
            "modest"       if abs(delta) < 2.0 else
            "meaningful"   if abs(delta) < 5.0 else
            "substantial"
        )
        direction = "improvement" if delta > 0 else "decline"
        traj_note = (
            f"shifting trajectory from {label_before} to {label_after}"
            if label_change else
            f"trajectory remains {label_after}"
        )
        lines += [
            "",
            f"PLAIN-ENGLISH INTERPRETATION: {magnitude.capitalize()} {direction} "
            f"of {abs(delta):.1f} points — {traj_note}.",
            "  Build your response around this sentence. Use the computed values above exactly.",
        ]

    # ── Per-feature impacts (combination mode) ────────────────────────────────
    individual_impacts = sim.get("individual_impacts", [])
    if individual_impacts:
        lines += ["", "Individual contributions (ranked by impact):"]
        for i, item in enumerate(individual_impacts, 1):
            feat = item["feature"]
            val  = item["value"]
            d    = item["delta"]
            disp = _FEAT_DISPLAY.get(feat, feat.replace("_", " "))
            if feat == "MedicationAdherence_enc":
                val_str = ADHERENCE_LABELS.get(float(val), str(val))
            else:
                rng     = _clinical_range_label(feat, float(val))
                val_str = f"{val}" + (f" [{rng}]" if rng else "")
            lines.append(f"  {i}. {disp} → {val_str}  (+{d:.2f} score points)")

    # ── SHAP weight shifts ────────────────────────────────────────────────────
    # Direction only — no raw magnitudes (prevents number leakage to patients).
    shap_shifts = sim.get("shap_shifts", [])
    if shap_shifts:
        lines += ["", "Features that drove the score change (model weight direction):"]
        for s in shap_shifts[:3]:
            display   = _FEAT_DISPLAY.get(s["feature"], s["feature"].replace("_", " "))
            direction = "increased its contribution" if s["shift"] > 0 else "decreased its contribution"
            lines.append(f"  {display}: {direction}")

    # ── Cascade warnings ──────────────────────────────────────────────────────
    cascade_notes = []
    for feat in changes:
        linked = FEATURE_CASCADES.get(feat, [])
        if linked:
            linked_names = [_FEAT_DISPLAY.get(f, f.replace("_", " ")) for f in linked]
            cascade_notes.append(
                f"  {_FEAT_DISPLAY.get(feat, feat)} is linked to "
                f"{', '.join(linked_names)} in real-world scenarios."
            )
    if cascade_notes:
        lines += ["", "Real-world cascade effects to mention:"]
        lines.extend(cascade_notes)

    # ── Safety note ───────────────────────────────────────────────────────────
    lines += [
        "",
        "SAFETY: This is a model simulation — not a clinical recommendation.",
        "Remind the patient to consult their care team before making any changes.",
        "[END SIMULATION]",
    ]

    return "\n".join(lines)
