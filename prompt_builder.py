import numpy as np
import pandas as pd

import reasoning
from utils import ADHERENCE_LABELS

TOP_N_SHAP = 5

# Use the comprehensive display-name mapping from reasoning.py as the base.
# Two Gender entries use "gender" here vs "sex" in reasoning.py — the overrides
# below preserve the exact wording that was already in the chatbot system prompt.
_DISPLAY = {
    **reasoning._DISPLAY,
    "Gender_Female": "Female gender",
    "Gender_Male":   "Male gender",
}

# One-sentence description of each phenotype for the patient overview and system prompt.
# These are shown before the conversation begins to orient the LLM.
_PHENOTYPE_SUMMARY = {
    "Cardiometabolic / Cholesterol Dominant": (
        "Your risk profile is primarily shaped by cholesterol and cardiovascular health — "
        "the model gives the greatest weight to cholesterol management for patients in your group. "
        "Evidence links elevated cholesterol with increased amyloid burden and cerebrovascular risk."
    ),
    "Cardiometabolic / BMI Dominant": (
        "Your risk profile is primarily shaped by body weight and metabolic health — "
        "BMI and adiposity-related factors have the strongest influence on the model's predictions for patients like you. "
        "Mid-life obesity is a well-established modifiable risk factor for cognitive decline."
    ),
    "Cardiometabolic / Heart Rate Dominant": (
        "Your risk profile is primarily shaped by heart rate and cardiovascular fitness — "
        "the model places the greatest weight on resting heart rate and cardiovascular function for patients in your group. "
        "Elevated resting heart rate is associated with reduced cerebral blood flow and accelerated cognitive ageing."
    ),
}


def _display_name(feature: str) -> str:
    return _DISPLAY.get(feature, feature.replace("_", " "))


def _get_prediction_shap(patient_id: str, shap_data: dict) -> tuple:
    """
    Return (shap_row, visit_row) for the visit that drove the patient's
    prediction — i.e. the last positional row in all_df for this patient,
    which matches how predictions_all.csv is built (groupby.last()).
    """
    all_df = shap_data["all_df"]
    shap_values = shap_data["shap_values"]

    indices = np.where(all_df["PatientID"].values == patient_id)[0]
    if len(indices) == 0:
        raise ValueError(f"Patient {patient_id!r} not found in SHAP data.")

    last_idx = indices[-1]
    return shap_values[last_idx], all_df.iloc[last_idx]


# CognitiveScore is kept in the model as an autoregressive anchor but must
# never surface in patient-facing SHAP explanations — it is not actionable
# and its prominence would obscure the clinically meaningful drivers.
_SHAP_DISPLAY_SUPPRESS = {
    "CognitiveScore",
    "CognitiveScore_prev",
    "CognitiveScore_delta",
    "CognitiveScore_roll3",
    "CognitiveScore_baseline",      # patient's cognitive reserve — used by model, not shown
    "CognitiveScore_from_baseline", # cumulative trend — used by model, not shown
}


def _top_shap_factors(shap_row: np.ndarray, feature_names: list, n: int = TOP_N_SHAP) -> list:
    ranked = sorted(enumerate(shap_row), key=lambda x: abs(x[1]), reverse=True)
    result = []
    for i, v in ranked:
        if feature_names[i] in _SHAP_DISPLAY_SUPPRESS:
            continue
        result.append({
            "feature":      feature_names[i],
            "display_name": _display_name(feature_names[i]),
            "shap_value":   float(v),
            "direction":    "increases" if v > 0 else "decreases",
        })
        if len(result) == n:
            break
    return result


def _extract_clinical_profile(visit_row: pd.Series) -> dict:
    """Pull the key measurable clinical values from the prediction visit row."""
    def _val(col, fmt=None):
        v = visit_row.get(col)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:{fmt}}" if fmt else v

    # One-hot fields — return the active category label
    def _onehot(prefix, options):
        for opt in options:
            col = f"{prefix}_{opt}"
            v = visit_row.get(col, 0)
            if v == 1 or v is True:
                return opt
        return "Unknown"

    diseases = [d for d in ["Alzheimer's", "Diabetes", "Parkinson's"]
                if visit_row.get(f"Disease_{d}", 0) == 1]
    med_hist = [h for h in ["Asthma", "Heart Disease", "Hypertension", "Stroke"]
                if visit_row.get(f"MedicalHistory_{h}", 0) == 1]

    return {
        "age":                  _val("Age", ".0f"),
        "gender":               _onehot("Gender", ["Female", "Male"]),
        "disease":              ", ".join(diseases) if diseases else "None documented",
        "medical_history":      ", ".join(med_hist) if med_hist else "None documented",
        "stage":                _val("Stage"),
        "bmi":                  _val("BMI", ".1f"),
        "cholesterol":          _val("Cholesterol", ".1f"),
        "heart_rate":           _val("HeartRate", ".0f"),
        "bp_systolic":          _val("BloodPressure_Systolic", ".0f"),
        "bp_diastolic":         _val("BloodPressure_Diastolic", ".0f"),
        "cognitive_score":      _val("CognitiveScore", ".1f"),
        "cognitive_baseline":   _val("CognitiveScore_baseline", ".1f"),
        "cognitive_delta":      _val("CognitiveScore_delta", ".2f"),
        "biomarker_score":      _val("BiomarkerScore", ".1f"),
        "medication_dose":      _val("MedicationDose", ".1f"),
        "medication_adherence": (
            ADHERENCE_LABELS.get(float(visit_row["MedicationAdherence_enc"]), "N/A")
            if "MedicationAdherence_enc" in visit_row
            and not (isinstance(visit_row["MedicationAdherence_enc"], float)
                     and np.isnan(visit_row["MedicationAdherence_enc"]))
            else _val("MedicationAdherence")
        ),
        "sleep_hours":          _val("SleepHours", ".1f"),
        "steps_per_day":        _val("StepsPerDay", ".0f"),
        "stress_level":         _val("StressLevel", ".1f"),
        "mood_score":           _val("MoodScore", ".1f"),
        "lifestyle":            _onehot("Lifestyle", ["Active", "Sedentary", "Non-Smoker",
                                                       "Smoker", "Occasional Drinker"]),
        "alcohol_use":          _onehot("AlcoholUse", ["High", "Moderate", "Unknown"]),
        "smoker":               _val("Smoker"),
        "support_system":       _val("SupportSystem"),
        "has_caregiver":        _val("HasCaregiver"),
        "employment":           _onehot("EmploymentStatus", ["Employed", "Retired", "Unemployed"]),
        "visit_index":          _val("visit_index", ".0f"),
        "visit_date":           str(visit_row.get("Date", "N/A")),
    }


def load_patient_context(
    patient_id: str,
    patient_data: pd.DataFrame,
    rag: dict,
    phenotype_stats: pd.DataFrame,
    shap_data: dict,
) -> dict:
    """
    Assemble all structured data for a patient into a single context dict.
    Raises ValueError if the patient is not found.
    """
    rows = patient_data[patient_data["PatientID"] == patient_id]
    if rows.empty:
        raise ValueError(f"Patient {patient_id!r} not found.")

    row = rows.iloc[0]
    phenotype = row["Phenotype_Simplified"]
    info = rag.get(phenotype, {})

    shap_row, visit_row = _get_prediction_shap(patient_id, shap_data)
    factors = _top_shap_factors(shap_row, shap_data["feature_names"])
    clinical_profile  = _extract_clinical_profile(visit_row)
    peer_reasoning = reasoning.compute_reasoning(
        patient_id    = patient_id,
        visit_row     = visit_row,
        shap_row      = shap_row,
        feature_names = shap_data["feature_names"],
        patient_data  = patient_data,
        all_df        = shap_data["all_df"],
        phenotype     = phenotype,
    )

    pred_score  = row.get("predicted_future_score")
    pred_label  = row.get("predicted_label")
    pred_change = row.get("predicted_change")

    ctx = {
        "patient_id":            patient_id,
        "phenotype":             phenotype,
        "phenotype_summary":     _PHENOTYPE_SUMMARY.get(phenotype, ""),
        "predicted_score":       float(pred_score)  if pd.notna(pred_score)  else None,
        "predicted_label":       str(pred_label)    if pd.notna(pred_label)  else "Unknown",
        "predicted_change":      float(pred_change) if pd.notna(pred_change) else None,
        "n_patients":            info.get("n_patients", "N/A"),
        "prevalence_pct":        info.get("prevalence_pct", "N/A"),
        "improving_pct":         float(info.get("improving_pct") or 0),
        "stable_pct":            float(info.get("stable_pct") or 0),
        "deteriorating_pct":     float(info.get("deteriorating_pct") or 0),
        "characteristics":       info.get("characteristics", "key health factors"),
        "recommendation":        info.get("recommendation", ""),
        "domain":                info.get("domain", ""),
        "dominant_feature":      info.get("dominant_feature", ""),
        "shap_factors":          factors,
        "clinical_profile":      clinical_profile,
        "reasoning":             peer_reasoning,
        "feature_trends":        reasoning.compute_feature_trends(
                                     patient_id, shap_data["all_df"]
                                 ),
    }
    ctx["clinical_reasoning"] = reasoning.generate_clinical_reasoning(ctx)
    return ctx


def build_patient_overview(ctx: dict) -> str:
    """
    Return a formatted patient overview card printed to the console before the
    conversation begins. Provides a fast at-a-glance summary for the clinician.
    """
    score_str = (
        f"{ctx['predicted_score']:.1f}"
        if ctx["predicted_score"] is not None
        else "N/A"
    )
    cp = ctx.get("clinical_profile", {})

    # SHAP factor lines with direction arrows and magnitude
    factor_lines = []
    for i, f in enumerate(ctx["shap_factors"], 1):
        sym = "▲" if f["direction"] == "increases" else "▼"
        factor_lines.append(
            f"    {i}. {f['display_name']:<40}  "
            f"{sym}  SHAP {abs(f['shap_value']):.3f}"
        )

    # Top actionable priority (if available) for at-a-glance guidance
    cr = ctx.get("clinical_reasoning", {})
    priorities = cr.get("actionable_priorities", [])
    top_priority = priorities[0] if priorities else ""
    conf_level = cr.get("confidence_level", "")
    conf_tag = f"  [{conf_level} confidence]" if conf_level else ""

    lines = [
        "=" * 65,
        f"  Patient Overview — {ctx['patient_id']}",
        "=" * 65,
        "",
        "  PREDICTION",
        f"    Predicted future score : {score_str}{conf_tag}",
        f"    Trajectory             : {ctx['predicted_label']}",
        "",
        "  PHENOTYPE GROUP",
        f"    {ctx['phenotype']}",
        f"    {ctx['n_patients']} patients ({ctx['prevalence_pct']}% of cohort)  |  "
        f"Improving {ctx['improving_pct']:.0f}%  "
        f"Stable {ctx['stable_pct']:.0f}%  "
        f"Deteriorating {ctx['deteriorating_pct']:.0f}%",
        f"    Primary driver: {ctx['characteristics']}",
        "",
        "  CLINICAL SNAPSHOT (prediction visit)",
        f"    Age {cp.get('age','?')} | {cp.get('gender','?')} | "
        f"Disease: {cp.get('disease','?')} | Stage: {cp.get('stage','?')}",
        f"    BMI {cp.get('bmi','?')} | Cholesterol {cp.get('cholesterol','?')} mg/dL | "
        f"HR {cp.get('heart_rate','?')} bpm | "
        f"BP {cp.get('bp_systolic','?')}/{cp.get('bp_diastolic','?')} mmHg",
        f"    Cognitive score {cp.get('cognitive_score','?')} "
        f"(baseline {cp.get('cognitive_baseline','?')}, "
        f"recent change {cp.get('cognitive_delta','?')})",
        f"    Sleep {cp.get('sleep_hours','?')}h | "
        f"Steps/day {cp.get('steps_per_day','?')} | "
        f"Stress {cp.get('stress_level','?')}/9 | "
        f"Adherence: {cp.get('medication_adherence','?')}",
        "",
        "  TOP SHAP FACTORS",
        *factor_lines,
    ]
    if top_priority:
        lines += [
            "",
            "  TOP ACTIONABLE PRIORITY",
            f"    {top_priority}",
        ]
    lines += [
        "",
        f"  Peer comparison: {ctx['reasoning'].get('n_peers', 0)} similar patients",
        "=" * 65,
    ]
    return "\n".join(lines)


def _format_reasoning_block(r: dict) -> str:
    """Format the reasoning dict as a compact, LLM-readable text block."""
    lines = [
        f"=== PEER COMPARISON & REASONING (N={r['n_peers']} similar patients) ===",
        "Use this section to explain WHY the prediction was made.",
        "Do NOT recite raw numbers unless asked. Synthesise into natural patient-friendly explanations.",
        r["overall_summary"],
        "",
    ]

    def _shap_importance_label(imp: float) -> str:
        if imp >= 0.3:   return "strongest driver"
        elif imp >= 0.15: return "notable factor"
        elif imp >= 0.05: return "moderate factor"
        else:             return "minor factor"

    def _pct_label(pct: int | None) -> str:
        if pct is None: return ""
        if pct >= 90:   return "among the highest of peers"
        if pct >= 75:   return "notably higher than most peers"
        if pct <= 10:   return "among the lowest of peers"
        if pct <= 25:   return "notably lower than most peers"
        return "near the peer group average"

    def _factor_lines(factors, label):
        if not factors:
            return
        lines.append(f"{label}:")
        for i, f in enumerate(factors, 1):
            pct  = f.get("percentile")
            prev = f.get("phenotype_prevalence_pct")
            if pct is not None:
                peer_s = f"  [{_pct_label(pct)}]"
            elif prev is not None:
                prev_word = "common" if prev >= 50 else ("uncommon" if prev < 25 else "present in some peers")
                peer_s = f"  [prevalence: {prev_word}]"
            else:
                peer_s = ""
            lines.append(
                f"  {i}. {f['display_name']}: patient={f['patient_value']}"
                f"{peer_s}"
                f"  |  {_shap_importance_label(f['shap_importance'])} ({f['shap_direction']} score)"
            )
            lines.append(f"     → {f['interpretation']}")
        lines.append("")

    _factor_lines(r["top_actionable_factors"],     "ACTIONABLE FACTORS (patient can influence)")
    _factor_lines(r["top_non_actionable_factors"], "NON-ACTIONABLE FACTORS (fixed characteristics)")

    lines.append("CAUTIONS:")
    for c in r["cautions"]:
        lines.append(f"  - {c}")

    return "\n".join(lines)


def _shap_label(imp: float) -> str:
    """Return qualitative importance label for use in data blocks."""
    if imp >= 0.3:    return "strongest driver"
    elif imp >= 0.15: return "notable factor"
    elif imp >= 0.05: return "moderate factor"
    else:             return "minor factor"


def _pct_position(pct: int | None) -> str:
    if pct is None: return ""
    if pct >= 90:   return "among the highest of peers"
    if pct >= 75:   return "notably higher than most peers"
    if pct <= 10:   return "among the lowest of peers"
    if pct <= 25:   return "notably lower than most peers"
    return "near the peer group average"


def _build_actionable_guidance(ctx: dict) -> str:
    """Build the recommendation-safe and off-limits sections from the reasoning dict."""
    from reasoning import _SUPPRESS_FROM_DISPLAY
    reasoning = ctx.get("reasoning", {})
    act = reasoning.get("top_actionable_factors", [])
    non = reasoning.get("top_non_actionable_factors", [])

    act_lines = []
    for i, f in enumerate(act, 1):
        if f.get("feature") in _SUPPRESS_FROM_DISPLAY:
            continue
        direction = "raises" if f["shap_direction"] == "increases" else "lowers"
        pct = f.get("percentile")
        if pct is not None:
            position = _pct_position(pct)
        else:
            position = f"present in {f.get('phenotype_prevalence_pct', '?')}% of peers"
        act_lines.append(
            f"  {i}. {f['display_name']} — {_shap_label(f['shap_importance'])}, "
            f"{direction} score | {position}"
        )

    non_names = ", ".join(f['display_name'] for f in non if f.get("feature") not in _SUPPRESS_FROM_DISPLAY) if non else "none in top factors"

    phenotype_rec = ctx.get("recommendation", "")
    rec_line = f"\nPhenotype-level recommendation: {phenotype_rec}" if phenotype_rec else ""

    return (
        "SAFE TO RECOMMEND (ranked by model influence):\n"
        + ("\n".join(act_lines) if act_lines else "  (none identified)")
        + f"\n\nNEVER RECOMMEND CHANGING: {non_names}, "
        + "age, sex, disease diagnosis, medical history, baseline cognition, "
        + "cognitive history, or time-in-study variables"
        + rec_line
    )


def _format_clinical_reasoning_block(cr: dict) -> str:
    """Format the clinical reasoning object as a structured LLM-readable block."""

    def _bullets(items: list[str]) -> str:
        return "\n".join(f"  • {item}" for item in items) if items else "  • (none identified)"

    def _numbered(items: list[str]) -> str:
        return "\n".join(f"  {i}. {item}" for i, item in enumerate(items, 1)) if items else "  1. (none identified)"

    return (
        "=== CLINICAL REASONING (synthesised — use this as your primary reference) ===\n"
        "Answer using this reasoning first. Use SHAP values and peer comparison data below\n"
        "only as supporting evidence. Do not simply list numbers — synthesise and explain.\n"
        "Distinguish clearly between model interpretation and established medical fact.\n"
        "State uncertainty when the confidence statement indicates it.\n"
        "\n"
        "MAJOR CONCERNS (factors the model associated with a lower predicted score):\n"
        + _bullets(cr["major_concerns"])
        + "\n\n"
        "PROTECTIVE FACTORS (factors associated with a better predicted outcome):\n"
        + _bullets(cr["protective_factors"])
        + "\n\n"
        "ACTIONABLE PRIORITIES (safe to recommend, ranked by model influence):\n"
        + _numbered(cr["actionable_priorities"])
        + "\n\n"
        "NON-ACTIONABLE CONTEXT (fixed characteristics — provide context only):\n"
        + _bullets(cr["non_actionable_factors"])
        + "\n\n"
        "PEER SUMMARY:\n"
        f"  {cr['peer_summary']}\n"
        "\n"
        "CONFIDENCE:\n"
        f"  Level: {cr.get('confidence_level', 'Moderate')}\n"
        f"  {cr['confidence_statement']}\n"
        "\n"
        "SUGGESTED FOLLOW-UPS (use these to guide conversation endings naturally):\n"
        + _bullets(cr["recommended_follow_up_questions"])
    )


def _build_clinical_narrative(ctx: dict) -> str:
    """
    Build a 3–4 sentence pre-synthesised plain-English narrative summarising
    the patient's situation.  This is the LLM's primary synthesis starting point
    — written so the LLM can build on interpretation, not raw tables.
    """
    cp = ctx.get("clinical_profile", {})
    cr = ctx.get("clinical_reasoning", {})
    r  = ctx.get("reasoning", {})
    t  = ctx.get("feature_trends", {})

    age         = cp.get("age", "?")
    gender      = cp.get("gender", "Unknown")
    disease     = cp.get("disease", "Unknown condition")
    stage       = cp.get("stage", "?")
    current_cog = cp.get("cognitive_score", "?")
    cog_delta   = cp.get("cognitive_delta", "?")
    pred_score  = ctx.get("predicted_score")
    pred_label  = ctx.get("predicted_label", "Unknown")
    pred_change = ctx.get("predicted_change")
    conf_level  = cr.get("confidence_level", "Moderate")

    score_str  = f"{pred_score:.1f}" if pred_score is not None else "unavailable"
    change_str = f"{pred_change:+.1f}" if pred_change is not None else ""

    traj_phrase = {
        "Improving":     "a positive trajectory",
        "Stable":        "a stable trajectory",
        "Deteriorating": "a concerning decline trajectory",
    }.get(pred_label, "an uncertain trajectory")

    # Sentence 1: demographic + current status
    try:
        delta_val = float(cog_delta)
        delta_note = f" (recent Δ {delta_val:+.1f})" if abs(delta_val) >= 0.5 else ""
    except (ValueError, TypeError):
        delta_note = ""
    sent1 = (
        f"This is a {age}-year-old {gender} patient with {disease} ({stage} stage), "
        f"currently at a cognitive score of {current_cog}{delta_note}."
    )

    # Sentence 2: prediction + confidence
    change_clause = f" ({change_str} from current)" if change_str else ""
    sent2 = (
        f"The model predicts their next score at {score_str}{change_clause}, "
        f"indicating {traj_phrase} — {conf_level.lower()} confidence."
    )

    # Sentence 3: top driver + trend
    all_factors = r.get("top_actionable_factors", []) + r.get("top_non_actionable_factors", [])
    sent3 = ""
    if all_factors:
        top    = all_factors[0]
        name   = top["display_name"]
        effect = "raises" if top["shap_direction"] == "increases" else "lowers"
        pct    = top.get("percentile")
        if pct is not None:
            if pct >= 90:   pct_note = " (among the highest of peers)"
            elif pct >= 75: pct_note = " (notably higher than most peers)"
            elif pct <= 10: pct_note = " (among the lowest of peers)"
            elif pct <= 25: pct_note = " (notably lower than most peers)"
            else:           pct_note = " (near the peer group average)"
        else:
            pct_note = ""

        trend_clause = ""
        feat_trend = t.get(top["feature"], {})
        if feat_trend and feat_trend["direction"] != "stable":
            mag_w = {"slight": "slightly", "moderate": "moderately", "notable": "notably"}[feat_trend["magnitude"]]
            trend_clause = f", and has been {mag_w} {feat_trend['direction']} over recent visits"

        sent3 = (
            f"The most influential model driver is {name}{pct_note} — "
            f"this {effect} the predicted score{trend_clause}."
        )

    # Sentence 4: top actionable priority (strip rank label to avoid "Key actionable focus: Top priority:")
    sent4 = ""
    priorities = cr.get("actionable_priorities", [])
    if priorities and "none identified" not in priorities[0].lower():
        # Priority sentences start with "Top priority: ...", "Second priority: ...", etc.
        # Prepend a cleaner label so the narrative reads naturally.
        import re as _re
        stripped = _re.sub(r"^(Top|Second|Third|Also worth discussing) priority:\s*", "", priorities[0])
        sent4 = f"Key actionable focus: {stripped}"

    parts = [s for s in [sent1, sent2, sent3, sent4] if s]
    return " ".join(parts)


def _build_trends_section(ctx: dict) -> str:
    """
    Format the feature-trends dict as a compact LLM-readable section.
    Returns an empty string if no trend data is available.
    """
    trends = ctx.get("feature_trends", {})
    if not trends:
        return ""

    lines = [
        "=== RECENT FEATURE TRENDS (direction over last visits) ===",
        "Use these to personalise explanations — e.g., 'your cholesterol has been rising.'",
    ]

    for feat, t in trends.items():
        display   = reasoning._DISPLAY.get(feat, feat.replace("_", " "))
        direction = t["direction"]
        magnitude = t["magnitude"]
        recent    = t.get("recent")
        first     = t.get("first")

        if feat == "MedicationAdherence_enc" and recent is not None:
            recent_str = ADHERENCE_LABELS.get(float(recent), str(recent))
            first_str  = ADHERENCE_LABELS.get(float(first), str(first)) if first is not None else "?"
            val_str = f"(now: {recent_str}, was: {first_str})"
        elif recent is not None:
            val_str = f"(now: {recent:.1f}, was: {first:.1f})"
        else:
            val_str = ""

        if direction == "stable":
            desc = f"stable {val_str}"
        else:
            mag_word = {"slight": "slightly", "moderate": "moderately", "notable": "notably"}[magnitude]
            desc = f"{mag_word} {direction} {val_str}"

        lines.append(f"  • {display}: {desc}")

    return "\n".join(lines)




def build_system_prompt(ctx: dict) -> str:
    """
    Build a concise, well-organised system prompt.

    Pre-computed clinical reasoning is presented as structured data.
    Instructions are minimal so a small model can reliably follow them.
    """
    score_str = (
        f"{ctx['predicted_score']:.1f}"
        if ctx["predicted_score"] is not None
        else "not available"
    )

    cr = ctx.get("clinical_reasoning", {})
    r  = ctx.get("reasoning", {})
    cp = ctx.get("clinical_profile", {})

    def _bullets(items: list[str]) -> str:
        return "\n".join(f"  • {s}" for s in items) if items else "  • (none identified)"

    def _numbered(items: list[str]) -> str:
        return "\n".join(f"  {i}. {s}" for i, s in enumerate(items, 1)) if items else "  1. (none identified)"

    clinical_block = (
        f"  Age {cp.get('age', '?')} | {cp.get('gender', '?')} | "
        f"Disease: {cp.get('disease', '?')} | Stage: {cp.get('stage', '?')}\n"
        f"  BMI {cp.get('bmi', '?')} | Cholesterol {cp.get('cholesterol', '?')} mg/dL | "
        f"HR {cp.get('heart_rate', '?')} bpm | "
        f"BP {cp.get('bp_systolic', '?')}/{cp.get('bp_diastolic', '?')} mmHg\n"
        f"  Cognitive: {cp.get('cognitive_score', '?')} "
        f"(baseline {cp.get('cognitive_baseline', '?')}, "
        f"recent change {cp.get('cognitive_delta', '?')})\n"
        f"  Sleep {cp.get('sleep_hours', '?')}h | "
        f"Steps/day {cp.get('steps_per_day', '?')} | "
        f"Stress {cp.get('stress_level', '?')}/9 | "
        f"Adherence: {cp.get('medication_adherence', '?')}\n"
        f"  Lifestyle: {cp.get('lifestyle', '?')} | "
        f"Smoker: {cp.get('smoker', '?')} | "
        f"Alcohol: {cp.get('alcohol_use', '?')}\n"
        f"  Medical history: {cp.get('medical_history', '?')} | "
        f"Support: {cp.get('support_system', '?')}"
    )

    narrative   = _build_clinical_narrative(ctx)
    trends_text = _build_trends_section(ctx)

    n_peers = r.get("n_peers", 0)
    peer_outcomes = (
        f"{ctx['improving_pct']:.0f}% improving | "
        f"{ctx['stable_pct']:.0f}% stable | "
        f"{ctx['deteriorating_pct']:.0f}% deteriorating"
    )

    phenotype_desc = ctx.get("phenotype_summary", "") or ctx.get("characteristics", "")

    follow_ups = cr.get("recommended_follow_up_questions", [])
    follow_up_text = "\n".join(f"  {i}. {q}" for i, q in enumerate(follow_ups, 1))

    trends_block = f"\nFEATURE TRENDS\n{trends_text}\n" if trends_text else ""

    return f"""You are a clinical AI assistant for patient {ctx['patient_id']}.
Your role: present and explain the pre-computed analysis below. You do NOT predict, diagnose, or generate information not present here.

RULES (apply to every response):
1. Base every statement on the sections below. If a question cannot be answered from this data, say so honestly.
2. MAJOR CONCERNS = factors associated with a LOWER predicted score. PROTECTIVE FACTORS = factors associated with a BETTER outcome. Never swap these framings — check the lists before writing.
3. Only recommend what is listed under RECOMMENDED ACTIONS. Never suggest changing: age, disease diagnosis, medical history, cognitive history, or visit-tracking variables.
4. Speak conversationally in second person ("your cholesterol", "your score"). Avoid jargon. Never quote raw model numbers — the qualitative language below is already written for you.
5. Write at least 3 sentences per response. For questions about factors, risks, or explanations, write at least 4.
6. End every response with one natural follow-up invitation drawn from SUGGESTED FOLLOW-UPS.
7. Never mention SHAP, XGBoost, visit_index, days_since_first, or any model internals.
8. Remember conversation history — refer back to earlier turns rather than repeating.

SIMULATION RULE (active only when a [WHAT-IF SIMULATION] block appears in the user message):
  These computed values are the source of truth. You are the interpreter — not the predictor.
  • Cite the exact original score, simulated score, and delta from the block.
  • Never add "probably", "likely", or "may" to an already-computed outcome.
  • Never use a simulated feature value as a starting point for further arithmetic.
  • If MINIMAL EFFECT is stated, say this feature is not a dominant model driver.
  • Do not contradict or extend the block using general medical knowledge.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATIENT SUMMARY
  {narrative}

PREDICTED OUTCOME
  Score      : {score_str}
  Trajectory : {ctx['predicted_label']}
  Confidence : {cr.get('confidence_level', 'Moderate')}
  {cr.get('confidence_statement', '').strip()}

MAJOR CONCERNS (factors the model associated with a lower predicted score — frame these as concerns)
{_bullets(cr.get('major_concerns', []))}

PROTECTIVE FACTORS (factors the model associated with a better outcome — frame these as strengths)
{_bullets(cr.get('protective_factors', []))}

RECOMMENDED ACTIONS (safe to discuss with patient, ranked by model influence)
{_numbered(cr.get('actionable_priorities', []))}

NON-ACTIONABLE CONTEXT (fixed characteristics — provide context only, do not recommend changing)
{_bullets(cr.get('non_actionable_factors', []))}

PEER CONTEXT — {n_peers} similar patients ({ctx['phenotype']} phenotype)
  {cr.get('peer_summary', '').strip()}
  Group outcomes: {peer_outcomes}
{trends_block}
PHENOTYPE: {ctx['phenotype']}
  {phenotype_desc}

BACKGROUND CLINICAL DATA (most recent prediction visit)
{clinical_block}

SUGGESTED FOLLOW-UPS
{follow_up_text}"""
