"""
Phenotype Generation Script
============================

Run after xgboost_and_shap.py:
    python generate_shap_phenotype.py

What this script does:
    1. Loads SHAP values from outputs/shap_values.pkl.
    2. Assigns each visit row a domain-based phenotype label by summing absolute
       SHAP importance across five clinical domains (Lifestyle, Cardiometabolic,
       Treatment, Disease, Demographic) and finding the strongest feature within
       the dominant domain.
    3. Deduplicates to one phenotype per patient using the modal label across visits.
    4. Merges phenotype groups with fewer than MERGE_THRESHOLD patients into the
       most similar large group using cosine similarity of domain-SHAP vectors.
    5. Joins with predictions_all.csv to compute per-phenotype outcome rates.
    6. Builds the RAG knowledge base (rag_phenotype_responses.json) used by the chatbot.

Outputs written to outputs/:
    baseline_with_shap_phenotypes.csv   — PatientID → Phenotype_Simplified
    phenotype_groups_simplified.csv     — per-phenotype counts and outcome rates
    rag_phenotype_responses.json        — chatbot knowledge base
"""

import os
import json
import pickle
import numpy as np
import pandas as pd

OUTPUT_DIR      = "outputs"
MERGE_THRESHOLD = 20  # phenotypes with fewer patients are merged into the closest large one


# =============================================================================
# DOMAIN DEFINITIONS
# =============================================================================

DOMAIN_FEATURES: dict[str, list[str]] = {
    "Lifestyle":       ["SleepHours", "StressLevel", "StepsPerDay"],
    "Cardiometabolic": ["BMI", "Cholesterol", "BloodPressure_Systolic", "HeartRate"],
    "Treatment":       ["MedicationDose", "MedicationAdherence"],
    "Disease":         ["BiomarkerScore"],
    "Demographic":     ["Age"],
}

DOMAIN_NAMES: list[str] = list(DOMAIN_FEATURES.keys())

FEATURE_LABELS: dict[str, str] = {
    "SleepHours":              "Sleep Dominant",
    "StressLevel":             "Stress Dominant",
    "StepsPerDay":             "Activity Dominant",
    "BMI":                     "BMI Dominant",
    "Cholesterol":             "Cholesterol Dominant",
    "BloodPressure_Systolic":  "Blood Pressure Dominant",
    "HeartRate":               "Heart Rate Dominant",
    "MedicationDose":          "Medication Dose Dominant",
    "MedicationAdherence":     "Adherence Dominant",
    "BiomarkerScore":          "Biomarker Dominant",
    "Age":                     "Age Dominant",
}

DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "Lifestyle":       "lifestyle factors including sleep quality, stress levels, and daily physical activity",
    "Cardiometabolic": "cardiometabolic risk factors such as cholesterol, BMI, blood pressure, and heart rate",
    "Treatment":       "treatment-related factors including medication adherence and dosage optimisation",
    "Disease":         "disease progression as measured by biomarker levels",
    "Demographic":     "age-related cognitive vulnerability",
}

FEATURE_CHARACTERISTICS: dict[str, str] = {
    "SleepHours":             "sleep duration and quality",
    "StressLevel":            "chronic stress and psychological burden",
    "StepsPerDay":            "daily physical activity and aerobic fitness",
    "BMI":                    "body weight, adiposity, and metabolic health",
    "Cholesterol":            "cholesterol levels and cardiovascular risk",
    "BloodPressure_Systolic": "systolic blood pressure and vascular health",
    "HeartRate":              "resting heart rate and cardiovascular conditioning",
    "MedicationDose":         "medication dosage and treatment intensity",
    "MedicationAdherence":    "consistency of medication use and treatment adherence",
    "BiomarkerScore":         "disease biomarker levels reflecting biological progression",
    "Age":                    "age-related cognitive vulnerability patterns",
}

# Clinical evidence context appended to chatbot responses per dominant feature.
FEATURE_EVIDENCE: dict[str, str] = {
    "SleepHours": (
        "Epidemiological evidence consistently links both short (<6 h) and long (>9 h) "
        "sleep with faster cognitive decline."
    ),
    "StressLevel": (
        "Chronic psychological stress elevates cortisol, which is associated with "
        "hippocampal atrophy and accelerated cognitive ageing."
    ),
    "StepsPerDay": (
        "Regular aerobic activity promotes BDNF release, improves cerebrovascular health, "
        "and is among the most robustly evidence-based modifiable factors for cognitive protection."
    ),
    "BMI": (
        "Mid-life obesity is an established risk factor for late-life dementia, partly "
        "mediated through vascular and metabolic pathways."
    ),
    "Cholesterol": (
        "Elevated LDL cholesterol in mid-life increases amyloid burden and cerebrovascular risk, "
        "both of which contribute to cognitive decline."
    ),
    "BloodPressure_Systolic": (
        "Hypertension is one of the strongest modifiable risk factors for vascular dementia "
        "and accelerates Alzheimer's pathology."
    ),
    "HeartRate": (
        "Elevated resting heart rate has been associated with reduced cerebral blood flow "
        "and worse cognitive outcomes, partly reflecting reduced cardiovascular fitness."
    ),
    "MedicationDose": (
        "Subtherapeutic or supratherapeutic dosing can affect disease control; "
        "optimal dosing is determined by the treating clinician based on individual factors."
    ),
    "MedicationAdherence": (
        "Non-adherence to prescribed treatment is a major driver of avoidable disease progression "
        "across Alzheimer's, Parkinson's, and diabetes."
    ),
    "BiomarkerScore": (
        "Biomarker trajectories reflect underlying disease activity; monitoring them closely "
        "allows earlier adjustment of the management plan."
    ),
    "Age": (
        "Age is the strongest non-modifiable risk factor for cognitive decline. "
        "While age cannot be changed, its effects can be partially offset by managing "
        "other modifiable risk factors."
    ),
}

DOMAIN_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "Lifestyle": {
        "SleepHours":  "prioritise consistent sleep schedules and aim for 7–9 hours per night; discuss sleep hygiene or sleep study with your doctor if sleep is disrupted",
        "StressLevel": "explore stress reduction techniques such as mindfulness, structured relaxation, or speaking with a mental health professional",
        "StepsPerDay": "work towards at least 7,500 steps per day through walking, cycling, or any aerobic activity you enjoy; gradual increases are safe and effective",
    },
    "Cardiometabolic": {
        "BMI":                    "work with your care team on a sustainable dietary and exercise plan to achieve and maintain a healthy BMI; avoid rapid or unsupported weight changes",
        "Cholesterol":            "discuss your cholesterol results with your doctor — dietary changes and statin therapy are both evidence-based options depending on your risk profile",
        "BloodPressure_Systolic": "monitor blood pressure regularly and follow your prescribed antihypertensive plan; sodium reduction and aerobic exercise also support BP control",
        "HeartRate":              "regular aerobic exercise lowers resting heart rate over time; discuss heart rate targets with your physician, especially if you have a cardiac history",
    },
    "Treatment": {
        "MedicationDose":      "discuss with your doctor whether your current dose is optimally calibrated — dose adjustments should always be medically supervised",
        "MedicationAdherence": "establish a consistent medication routine using pill organisers, phone reminders, or pharmacy blister packs; never adjust doses without consulting your doctor",
    },
    "Disease": {
        "BiomarkerScore": "attend all scheduled monitoring appointments so your care team can track biomarker trends and adjust your management plan proactively",
    },
    "Demographic": {
        "Age": "engage in regular cognitive stimulation (reading, puzzles, learning new skills), maintain social connections, and ensure all modifiable risk factors are well-managed",
    },
}


# =============================================================================
# PHENOTYPE FUNCTIONS
# =============================================================================

def get_domain_and_base(fname: str) -> tuple[str | None, str | None]:
    """Map a (possibly derived) feature name to its clinical domain and base feature."""
    for domain, bases in DOMAIN_FEATURES.items():
        for base in bases:
            if fname == base or fname.startswith(base + "_"):
                return domain, base
    return None, None


def create_domain_phenotype(shap_row: np.ndarray, feature_names: list[str]) -> tuple[str, dict]:
    """
    Assign a phenotype label and compute domain SHAP totals for one visit row.

    Returns
    -------
    phenotype : str
        e.g. "Cardiometabolic / Cholesterol Dominant"
    full_totals : dict
        Total absolute SHAP importance per domain (all five domains present).
    """
    domain_data: dict[str, list] = {d: [] for d in DOMAIN_FEATURES}

    for i, fname in enumerate(feature_names):
        domain, base = get_domain_and_base(fname)
        if domain is not None:
            domain_data[domain].append((i, base, abs(shap_row[i])))

    domain_totals = {
        d: sum(x[2] for x in items)
        for d, items in domain_data.items()
        if items
    }
    full_totals = {d: domain_totals.get(d, 0.0) for d in DOMAIN_NAMES}

    if not domain_totals:
        return "Unknown / Unknown", full_totals

    dominant_domain = max(domain_totals, key=domain_totals.get)
    strongest = max(domain_data[dominant_domain], key=lambda x: x[2])
    label = FEATURE_LABELS.get(strongest[1], f"{strongest[1]} Dominant")

    return f"{dominant_domain} / {label}", full_totals


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two domain-SHAP vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def merge_small_phenotypes(df: pd.DataFrame, domain_cols: list[str], threshold: int) -> pd.DataFrame:
    """
    Iteratively absorb phenotypes below `threshold` patients into the most
    similar large phenotype by cosine similarity of mean domain-SHAP vectors.
    Merges smallest-first to avoid premature absorption.
    """
    df = df.copy()
    print(f"\nMerging phenotypes with fewer than {threshold} patients...")

    iteration = 0
    while True:
        sizes = df.groupby("Phenotype_Simplified").size()
        small = sizes[sizes < threshold].sort_values()

        if small.empty:
            break

        iteration += 1
        large = sizes[sizes >= threshold]

        if large.empty:
            print("  Warning: all phenotypes are below threshold — skipping merge.")
            break

        large_vecs = {
            ph: df.loc[df["Phenotype_Simplified"] == ph, domain_cols].mean().values
            for ph in large.index
        }

        for ph in small.index:
            mask      = df["Phenotype_Simplified"] == ph
            small_vec = df.loc[mask, domain_cols].mean().values

            best_ph  = max(large_vecs, key=lambda t: cosine_sim(small_vec, large_vecs[t]))
            best_sim = cosine_sim(small_vec, large_vecs[best_ph])

            print(
                f"  [{iteration}] '{ph}' ({int(sizes[ph])} patients)"
                f" → '{best_ph}'"
                f" (cosine similarity: {best_sim:.3f})"
            )
            df.loc[mask, "Phenotype_Simplified"] = best_ph

    return df


def build_rag_entry(row: pd.Series) -> dict:
    """
    Build the RAG knowledge-base entry for one phenotype.
    Includes pre-written chatbot responses and evidence context.
    """
    phenotype = row["Phenotype_Simplified"]
    parts     = phenotype.split(" / ", 1)
    domain    = parts[0] if len(parts) == 2 else "Unknown"
    feat_lbl  = parts[1] if len(parts) == 2 else phenotype

    dominant_feature = next(
        (k for k, v in FEATURE_LABELS.items() if v == feat_lbl), None
    )

    domain_desc  = DOMAIN_DESCRIPTIONS.get(domain, "key health factors")
    feature_chars = FEATURE_CHARACTERISTICS.get(dominant_feature, feat_lbl.lower())
    evidence     = FEATURE_EVIDENCE.get(dominant_feature, "")

    recommendation = ""
    if dominant_feature and domain in DOMAIN_RECOMMENDATIONS:
        recommendation = DOMAIN_RECOMMENDATIONS[domain].get(dominant_feature, "")
    if not recommendation and domain in DOMAIN_RECOMMENDATIONS:
        recommendation = next(iter(DOMAIN_RECOMMENDATIONS[domain].values()), "")

    improving    = float(row.get("Improving_%") or 0)
    stable       = float(row.get("Stable_%") or 0)
    deteriorating = float(row.get("Deteriorating_%") or 0)
    n   = int(row["N_Patients"])
    pct = float(row["Pct_of_Population"])

    # Build a concise majority-outcome descriptor for use in chatbot prose
    if improving > deteriorating and improving > stable:
        outcome_summary = f"most patients ({improving:.0f}%) in this group are on an improving trajectory"
    elif deteriorating > improving and deteriorating > stable:
        outcome_summary = f"a notable proportion ({deteriorating:.0f}%) are currently deteriorating — making active management especially important"
    else:
        outcome_summary = f"outcomes are mixed: {improving:.0f}% improving, {stable:.0f}% stable, {deteriorating:.0f}% deteriorating"

    return {
        "phenotype":          phenotype,
        "domain":             domain,
        "dominant_feature":   dominant_feature,
        "n_patients":         n,
        "prevalence_pct":     pct,
        "improving_pct":      improving,
        "stable_pct":         stable,
        "deteriorating_pct":  deteriorating,
        "characteristics":    feature_chars,
        "recommendation":     recommendation,
        "evidence_context":   evidence,
        "chatbot_responses": {
            "which_cluster": (
                f"You belong to the '{phenotype}' phenotype, which represents "
                f"{pct:.1f}% of all patients in this dataset ({n} individuals). "
                f"This group is characterised by {domain_desc} being the primary driver "
                f"of cognitive trajectory predictions."
            ),
            "what_defines": (
                f"Within the {domain} domain, {feature_chars} stands out as the single "
                f"strongest contributor to how the model predicts your cognitive trajectory. "
                f"Looking at outcomes in your group: {outcome_summary}. "
                f"Clinically, {evidence}" if evidence else
                f"Within the {domain} domain, {feature_chars} stands out as the single "
                f"strongest contributor to how the model predicts your cognitive trajectory. "
                f"Looking at outcomes in your group: {outcome_summary}."
            ),
            "what_to_focus": (
                f"For patients in your phenotype, the highest-leverage area is {feature_chars}. "
                f"The evidence-informed suggestion is to {recommendation}. "
                f"This is the factor the model considers most influential for patients like you — "
                f"but any changes should be discussed with your care team first."
            ),
            "what_improves": (
                f"Among the {n} patients in the '{phenotype}' group, {improving:.0f}% showed "
                f"cognitive improvement in their predicted trajectory. "
                f"Patients who improved most consistently tended to show better management of "
                f"{feature_chars}. "
                f"The model cannot establish causation, but this pattern supports prioritising "
                f"this area with your clinician."
            ),
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load SHAP data ────────────────────────────────────────────────────────
    shap_path = os.path.join(OUTPUT_DIR, "shap_values.pkl")
    print(f"Loading SHAP data from {shap_path}...")
    with open(shap_path, "rb") as f:
        shap_data = pickle.load(f)

    shap_values   = shap_data["shap_values"]
    feature_names = shap_data["feature_names"]
    all_df        = shap_data["all_df"].copy()

    print(f"Loaded {len(all_df)} rows, {all_df['PatientID'].nunique()} unique patients")

    # ── Assign phenotype to every visit row ───────────────────────────────────
    print("\nAssigning domain-based phenotypes...")
    results         = [create_domain_phenotype(row, feature_names) for row in shap_values]
    phenotypes_raw  = [r[0] for r in results]
    domain_vecs_raw = [r[1] for r in results]

    all_df["Phenotype_Simplified"] = phenotypes_raw

    DOMAIN_COLS = [f"_domain_{d}" for d in DOMAIN_NAMES]
    for d, col in zip(DOMAIN_NAMES, DOMAIN_COLS):
        all_df[col] = [v[d] for v in domain_vecs_raw]

    # ── Deduplicate: modal phenotype + mean domain vector per patient ─────────
    patient_phenotypes = (
        all_df.groupby("PatientID")["Phenotype_Simplified"]
        .agg(lambda x: x.mode().iloc[0])
        .reset_index()
    )
    patient_domain_vecs = (
        all_df.groupby("PatientID")[DOMAIN_COLS]
        .mean()
        .reset_index()
    )
    patient_phenotypes = patient_phenotypes.merge(patient_domain_vecs, on="PatientID", how="left")

    print("\nInitial phenotype counts:")
    for ph, n in patient_phenotypes["Phenotype_Simplified"].value_counts().items():
        print(f"  {ph}: {n}")

    # ── Merge small phenotypes ────────────────────────────────────────────────
    patient_phenotypes = merge_small_phenotypes(patient_phenotypes, DOMAIN_COLS, MERGE_THRESHOLD)

    # ── Load outcomes and compute per-phenotype rates ─────────────────────────
    pred_path = os.path.join(OUTPUT_DIR, "predictions_all.csv")
    if os.path.exists(pred_path):
        preds = pd.read_csv(pred_path)
        patient_phenotypes = patient_phenotypes.merge(
            preds[["PatientID", "predicted_label"]], on="PatientID", how="left"
        )

    n_unique_patients = patient_phenotypes["PatientID"].nunique()

    profiles = []
    for phenotype, group in patient_phenotypes.groupby("Phenotype_Simplified"):
        n          = len(group)
        has_labels = "predicted_label" in group.columns
        profiles.append({
            "Phenotype_Simplified": phenotype,
            "N_Patients":           n,
            "Pct_of_Population":    round(n / n_unique_patients * 100, 1),
            "Improving_%":    round((group["predicted_label"] == "Improving").mean() * 100, 1) if has_labels else np.nan,
            "Stable_%":       round((group["predicted_label"] == "Stable").mean() * 100, 1)    if has_labels else np.nan,
            "Deteriorating_%":round((group["predicted_label"] == "Deteriorating").mean() * 100, 1) if has_labels else np.nan,
        })

    profiles_df = (
        pd.DataFrame(profiles)
        .sort_values("N_Patients", ascending=False)
        .reset_index(drop=True)
    )
    final_sizes = profiles_df["N_Patients"]

    print(f"\n{'='*50}")
    print("FINAL PHENOTYPE SUMMARY")
    print(f"{'='*50}")
    print(f"Number of final phenotypes : {len(profiles_df)}")
    print(f"Minimum phenotype size     : {final_sizes.min()}")
    print(f"Maximum phenotype size     : {final_sizes.max()}")
    print("\nFinal phenotype breakdown:")
    for _, row in profiles_df.iterrows():
        print(
            f"  {row['Phenotype_Simplified']}: {int(row['N_Patients'])} patients "
            f"({row['Pct_of_Population']}%)  |  "
            f"Improving {row['Improving_%']:.0f}%  "
            f"Stable {row['Stable_%']:.0f}%  "
            f"Deteriorating {row['Deteriorating_%']:.0f}%"
        )
    print(f"{'='*50}")

    # ── Save phenotype group summary ──────────────────────────────────────────
    profiles_df.to_csv(os.path.join(OUTPUT_DIR, "phenotype_groups_simplified.csv"), index=False)

    # ── Build and save RAG knowledge base ─────────────────────────────────────
    rag = {
        row["Phenotype_Simplified"]: build_rag_entry(row)
        for _, row in profiles_df.iterrows()
    }
    with open(os.path.join(OUTPUT_DIR, "rag_phenotype_responses.json"), "w") as f:
        json.dump(rag, f, indent=2)

    # ── Save patient phenotype assignments ────────────────────────────────────
    patient_phenotypes[["PatientID", "Phenotype_Simplified"]].to_csv(
        os.path.join(OUTPUT_DIR, "baseline_with_shap_phenotypes.csv"), index=False
    )

    print("\nGenerated outputs:")
    print(f"  outputs/baseline_with_shap_phenotypes.csv  ({n_unique_patients} patients)")
    print(f"  outputs/phenotype_groups_simplified.csv    ({len(profiles_df)} phenotypes)")
    print(f"  outputs/rag_phenotype_responses.json       ({len(rag)} entries)")


if __name__ == "__main__":
    main()
