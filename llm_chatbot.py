#!/usr/bin/env python3
"""
LLM Chatbot — Main Entry Point
================================

Run this script to start the interactive patient chatbot:
    python llm_chatbot.py

Workflow per session:
    1. Load all data (phenotypes, predictions, SHAP values, RAG knowledge base).
    2. Prompt the user for a Patient ID.
    3. Build the full LLM system prompt for that patient via prompt_builder.
    4. Enter a conversation loop:
        - Pass the full conversation history to llm_backend.call_llm().
    5. Type "new" to switch patient or "quit" to exit.

Dependencies (must be run first):
    xgboost_and_shap.py       → produces predictions_all.csv, shap_values.pkl
    generate_shap_phenotype.py → produces baseline_with_shap_phenotypes.csv,
                                  phenotype_groups_simplified.csv,
                                  rag_phenotype_responses.json
"""

import os
import json
import pickle
import pandas as pd

from prompt_builder import load_patient_context, build_patient_overview, build_system_prompt
from llm_backend import call_llm
import simulation_engine as sim_engine

OUTPUT_DIR = "outputs"

# ── Conversational intro ────────────────────────────────────────────────────────

_INTRO_REQUEST = (
    "Open our session with a warm, conversational briefing on this patient. "
    "Write 2–3 short paragraphs (not bullet points). Cover: "
    "(1) the predicted future cognitive score and trajectory, framed in plain language; "
    "(2) the most important factor the model identified and what it means in practice; "
    "(3) one concrete suggestion for what we might explore together. "
    "Use second person ('your patient', 'they'). "
    "Avoid raw numbers unless they add clear meaning. "
    "End with a natural invitation for questions — do not list suggested questions explicitly. "
    "Keep the total response under 150 words. Be warm, clear, and clinically grounded."
)

_FOLLOW_UP_REMINDER = (
    "\n\n[INSTRUCTION REMINDER: End your response with a natural "
    "follow-up invitation sentence. For example: 'Would you like "
    "to know more about any of these factors?' or 'If you'd like "
    "to explore this further, just ask.']"
)

_SUGGESTED_QUESTIONS = [
    "Why did the model predict this?",
    "What are the biggest risk factors?",
    "What should I improve first?",
    "Compare me with similar patients.",
    "Generate a patient summary.",
    "What if my cholesterol dropped to 180?",
    "What's the best single change I could make?",
]


def _generate_intro(system_prompt: str) -> tuple[str, list]:
    """
    Ask the LLM to generate a short conversational opening for the patient session.
    Returns (intro_text, initial_history) so the intro is part of the conversation
    context for all follow-up questions.

    The verbose _INTRO_REQUEST is used only for generation and is NOT stored in the
    persistent history — doing so would teach the model that user messages are
    structured internal instructions, corrupting future conversation patterns.
    A short placeholder is stored instead.
    """
    seed_history = [{"role": "user", "content": _INTRO_REQUEST}]
    intro = call_llm(system_prompt, seed_history)
    history = [
        {"role": "user",      "content": "Please introduce this patient."},
        {"role": "assistant", "content": intro},
    ]
    return intro, history


def _load_data():
    """
    Load all data files produced by the offline pipeline into memory.

    Returns
    -------
    patient_data   : DataFrame — one row per patient: phenotype + predictions merged.
    rag            : dict — phenotype knowledge base from rag_phenotype_responses.json.
    phenotype_stats: DataFrame — phenotype group summary statistics.
    shap_data      : dict — SHAP values, feature names, X_all, all_df.
    """
    baseline = pd.read_csv(os.path.join(OUTPUT_DIR, "baseline_with_shap_phenotypes.csv"))
    predictions = pd.read_csv(os.path.join(OUTPUT_DIR, "predictions_all.csv"))
    phenotype_stats = pd.read_csv(os.path.join(OUTPUT_DIR, "phenotype_groups_simplified.csv"))

    with open(os.path.join(OUTPUT_DIR, "rag_phenotype_responses.json")) as f:
        rag = json.load(f)

    with open(os.path.join(OUTPUT_DIR, "shap_values.pkl"), "rb") as f:
        shap_data = pickle.load(f)

    patient_data = baseline.merge(
        predictions[["PatientID", "predicted_future_score", "predicted_change", "predicted_label"]],
        on="PatientID",
        how="left",
    )

    return patient_data, rag, phenotype_stats, shap_data



def main():
    print("Loading data...")
    patient_data, rag, phenotype_stats, shap_data = _load_data()
    n_patients = patient_data["PatientID"].nunique()
    print(f"Ready — {n_patients} patients loaded.")
    print("Type 'quit' at any prompt to exit.\n")

    engine = sim_engine.SimulationEngine()

    while True:
        patient_id = input("Patient ID: ").strip()
        if patient_id.lower() in ("quit", "q", "exit"):
            break
        if not patient_id:
            continue

        try:
            ctx = load_patient_context(patient_id, patient_data, rag, phenotype_stats, shap_data)
        except ValueError as exc:
            print(f"  {exc}\n")
            continue

        system_prompt = build_system_prompt(ctx)

        print()
        print(build_patient_overview(ctx))
        print()

        # ── Conversational intro (LLM-generated) ─────────────────────────────
        print("-" * 50)
        print()
        _, history = _generate_intro(system_prompt)
        print()
        print("-" * 50)
        print()

        # ── Proactive question guide ──────────────────────────────────────────
        print("You could ask me things like:\n")
        for q in _SUGGESTED_QUESTIONS:
            print(f"  • {q}")
        print()
        print("Or simply ask your own question.")
        print()
        print("  (Type 'new' to switch patient  |  'quit' to exit)\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return

            if not user_input:
                continue

            if user_input.lower() in ("quit", "q", "exit"):
                return

            if user_input.lower() in ("new", "change", "switch"):
                print()
                break

            # --- build the two versions of the user turn ---
            # history_content : clean text stored permanently in history.
            #   Contains the user's question and (if a simulation ran) the
            #   simulation result block for follow-up context.
            #   Never contains ephemeral instructions or reminders.
            # call_content    : what the LLM actually receives this turn.
            #   Adds the simulation-mode trigger and the follow-up reminder on
            #   top of history_content.  Discarded after the call.

            history_content = user_input
            call_content    = user_input

            if sim_engine.is_whatif_question(user_input):
                try:
                    changes = sim_engine.parse_whatif_question(user_input, ctx, shap_data)
                    if changes == "__best_single__":
                        result = engine.find_best_single_change(patient_id, shap_data)
                    elif changes == "__best_combination__":
                        result = engine.find_best_combination(patient_id, shap_data)
                    else:
                        result = engine.simulate(patient_id, changes, shap_data)
                    sim_block = sim_engine.format_simulation_result(result)

                    # History: simulation results + original question (no mode trigger)
                    history_content = f"{sim_block}\n\nPatient question: {user_input}"

                    # LLM call: same content + explicit mode trigger
                    call_content = (
                        f"{sim_block}\n\n"
                        "[SIMULATION MODE ACTIVE]\n"
                        "The block above contains COMPUTED results from simulation_engine.py.\n"
                        "These are the source of truth. You MUST act as an interpreter of "
                        "these exact results only.\n"
                        "Do NOT estimate, predict, or generate new numbers.\n"
                        "Use only the before/after scores, delta, and feature values shown above.\n"
                        "Do NOT treat any simulated feature value as a new baseline for "
                        "further arithmetic.\n\n"
                        f"Patient question: {user_input}"
                    )
                except Exception as exc:
                    print(f"  [Simulation error: {exc}]")

            # Persist the clean turn in history (no reminder, no mode trigger)
            history.append({"role": "user", "content": history_content})

            # Build a one-shot call history: all history except the just-appended
            # entry, then replace it with call_content + the follow-up reminder.
            # This means the reminder is sent to the LLM this turn but never
            # accumulates in history, preventing pattern corruption over many turns.
            call_history = history[:-1] + [{
                "role": "user",
                "content": call_content + _FOLLOW_UP_REMINDER,
            }]

            print("Assistant: ", end="", flush=True)
            reply = call_llm(system_prompt, call_history)
            print()

            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
