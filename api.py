"""
Healthcare Explainable AI — REST API
=====================================
Exposes all backend functionality as JSON endpoints consumed by the
React frontend.  The LLM chat endpoint streams tokens via SSE.

Run:
    python api.py
"""

import os
import json
import pickle
import queue
import threading
import traceback

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context
from flask_cors import CORS

from prompt_builder import load_patient_context, build_patient_overview, build_system_prompt
from llm_backend import call_llm, OLLAMA_URL, OLLAMA_MODEL
import simulation_engine as sim_engine

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR   = "outputs"
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend", "dist")

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

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="/")
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---------------------------------------------------------------------------
# Data loading (once at startup)
# ---------------------------------------------------------------------------

_data: dict = {}

def _load_data():
    baseline      = pd.read_csv(os.path.join(OUTPUT_DIR, "baseline_with_shap_phenotypes.csv"))
    predictions   = pd.read_csv(os.path.join(OUTPUT_DIR, "predictions_all.csv"))
    phenotype_stats = pd.read_csv(os.path.join(OUTPUT_DIR, "phenotype_groups_simplified.csv"))

    with open(os.path.join(OUTPUT_DIR, "rag_phenotype_responses.json")) as f:
        rag = json.load(f)

    with open(os.path.join(OUTPUT_DIR, "shap_values.pkl"), "rb") as f:
        shap_data = pickle.load(f)

    patient_data = baseline.merge(
        predictions[["PatientID", "predicted_future_score", "predicted_change", "predicted_label"]],
        on="PatientID", how="left",
    )

    feature_importance_csv = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    feature_importance = pd.read_csv(feature_importance_csv).to_dict(orient="records") if os.path.exists(feature_importance_csv) else []

    _data["patient_data"]     = patient_data
    _data["rag"]              = rag
    _data["phenotype_stats"]  = phenotype_stats
    _data["shap_data"]        = shap_data
    _data["feature_importance"] = feature_importance
    _data["engine"]           = sim_engine.SimulationEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ctx(patient_id: str) -> dict:
    """Build and return patient context dict; raises ValueError if not found."""
    return load_patient_context(
        patient_id,
        _data["patient_data"],
        _data["rag"],
        _data["phenotype_stats"],
        _data["shap_data"],
    )


def _serialize(obj):
    """JSON-safe conversion for numpy / pandas types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    raise TypeError(f"Not serializable: {type(obj)}")


def _deep_serialize(obj):
    """Recursively convert numpy/pandas types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deep_serialize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Series):
        return {k: _deep_serialize(v) for k, v in obj.items()}
    if isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    return obj


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, default=_serialize)}\n\n"


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "ollama_url": OLLAMA_URL,
        "ollama_model": OLLAMA_MODEL,
        "patients_loaded": _data.get("patient_data", pd.DataFrame()).shape[0] if "patient_data" in _data else 0,
    })


@app.route("/api/patients")
def get_patients():
    """Return sorted list of all patient IDs."""
    pd_df = _data["patient_data"]
    ids = sorted(pd_df["PatientID"].unique().tolist())
    return jsonify(ids)


@app.route("/api/patient/<patient_id>")
def get_patient(patient_id: str):
    """
    Return full patient context as JSON.
    This is the primary data source for the Overview, Phenotype, and Peers pages.
    """
    try:
        ctx = _ctx(patient_id.strip())
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    # Enrich with peer stats from phenotype_stats
    ps = _data["phenotype_stats"]
    phenotype = ctx["phenotype"]
    peer_row = ps[ps["Phenotype_Simplified"] == phenotype]
    peer_stats = peer_row.iloc[0].to_dict() if not peer_row.empty else {}

    # Feature trends in serializable form
    trends = {
        k: {
            "direction": v["direction"],
            "magnitude": v["magnitude"],
            "n_visits":  v["n_visits"],
            "recent":    float(v["recent"]) if v.get("recent") is not None else None,
            "first":     float(v["first"])  if v.get("first")  is not None else None,
        }
        for k, v in ctx.get("feature_trends", {}).items()
    }

    cr = ctx.get("clinical_reasoning", {})

    payload = {
        "patient_id":          ctx["patient_id"],
        "phenotype":           ctx["phenotype"],
        "phenotype_summary":   ctx["phenotype_summary"],
        "predicted_score":     ctx["predicted_score"],
        "predicted_label":     ctx["predicted_label"],
        "predicted_change":    ctx["predicted_change"],
        "n_patients":          ctx["n_patients"],
        "prevalence_pct":      ctx["prevalence_pct"],
        "improving_pct":       ctx["improving_pct"],
        "stable_pct":          ctx["stable_pct"],
        "deteriorating_pct":   ctx["deteriorating_pct"],
        "characteristics":     ctx["characteristics"],
        "recommendation":      ctx["recommendation"],
        "domain":              ctx["domain"],
        "dominant_feature":    ctx["dominant_feature"],
        "shap_factors":        ctx["shap_factors"],
        "clinical_profile":    ctx["clinical_profile"],
        "reasoning":           {
            "n_peers":                    ctx["reasoning"].get("n_peers", 0),
            "overall_summary":            ctx["reasoning"].get("overall_summary", ""),
            "top_actionable_factors":     ctx["reasoning"].get("top_actionable_factors", []),
            "top_non_actionable_factors": ctx["reasoning"].get("top_non_actionable_factors", []),
            "cautions":                   ctx["reasoning"].get("cautions", []),
        },
        "clinical_reasoning":  {
            "major_concerns":                  cr.get("major_concerns", []),
            "protective_factors":              cr.get("protective_factors", []),
            "actionable_priorities":           cr.get("actionable_priorities", []),
            "non_actionable_factors":          cr.get("non_actionable_factors", []),
            "peer_summary":                    cr.get("peer_summary", ""),
            "confidence_level":                cr.get("confidence_level", "Moderate"),
            "confidence_statement":            cr.get("confidence_statement", ""),
            "recommended_follow_up_questions": cr.get("recommended_follow_up_questions", []),
        },
        "feature_trends": trends,
        "peer_stats":     {k: (float(v) if isinstance(v, (int, float, np.number)) else v)
                           for k, v in peer_stats.items()},
        "rag_info": _data["rag"].get(phenotype, {}),
    }

    # Cognitive score history across all visits
    all_df = _data["shap_data"]["all_df"]
    pat_visits = (
        all_df[all_df["PatientID"] == patient_id.strip()]
        .sort_values("Date")
    )
    cog_history = []
    for _, vr in pat_visits.iterrows():
        cs = vr.get("CognitiveScore")
        if cs is not None and not (isinstance(cs, float) and cs != cs):
            cog_history.append({
                "date":        str(vr["Date"]),
                "score":       float(cs),
                "visit_index": int(vr["visit_index"]) if pd.notna(vr.get("visit_index")) else None,
            })
    payload["cognitive_score_history"] = cog_history

    return jsonify(_deep_serialize(payload))


@app.route("/api/patient/<patient_id>/intro")
def patient_intro(patient_id: str):
    """
    SSE stream: generate and stream the conversational intro for a patient.
    Events:
      {type: "token",    content: "..."}
      {type: "metadata", clean_content: "..."}   # message to store in history
      {type: "done"}
      {type: "error",    message: "..."}
    """
    try:
        ctx = _ctx(patient_id.strip())
    except ValueError as e:
        def err():
            yield _sse({"type": "error", "message": str(e)})
        return Response(stream_with_context(err()), mimetype="text/event-stream")

    system_prompt = build_system_prompt(ctx)

    def generate():
        token_q: queue.Queue = queue.Queue()
        full_tokens: list[str] = []

        def _call():
            # Custom streaming: collect tokens and put them in queue
            import requests as _req
            import json as _json
            messages = [
                {"role": "system",    "content": system_prompt},
                {"role": "user",      "content": _INTRO_REQUEST},
            ]
            try:
                resp = _req.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "stop":        ["\nYou:", "\nUser:", "\nPatient:"],
                            "temperature": 0.1,
                            "repeat_penalty": 1.15,
                            "top_p": 0.9,
                        },
                    },
                    timeout=120,
                    stream=True,
                )
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    chunk = _json.loads(raw_line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full_tokens.append(token)
                        token_q.put(("token", token))
                    if chunk.get("done"):
                        break
            except Exception as exc:
                token_q.put(("error", str(exc)))
            finally:
                token_q.put(("done", None))

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()

        while True:
            kind, payload = token_q.get()
            if kind == "token":
                yield _sse({"type": "token", "content": payload})
            elif kind == "error":
                yield _sse({"type": "error", "message": payload})
                return
            elif kind == "done":
                full_text = "".join(full_tokens)
                yield _sse({"type": "metadata", "clean_content": "Please introduce this patient.", "response": full_text})
                yield _sse({"type": "done"})
                return

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/patient/<patient_id>/chat", methods=["POST"])
def chat(patient_id: str):
    """
    SSE stream: process a chat turn and stream the response.

    Request JSON:
      {
        "message":  "user message text",
        "history":  [{"role": "user"|"assistant", "content": "..."}]
      }

    SSE events:
      {type: "sim_result",  result: {...}}           # if simulation ran
      {type: "token",       content: "..."}
      {type: "metadata",    clean_content: "..."}    # what to store in history
      {type: "done"}
      {type: "error",       message: "..."}
    """
    body = request.get_json(force=True, silent=True) or {}
    user_message = body.get("message", "").strip()
    history      = body.get("history", [])

    if not user_message:
        def empty():
            yield _sse({"type": "error", "message": "Empty message"})
        return Response(stream_with_context(empty()), mimetype="text/event-stream")

    try:
        ctx = _ctx(patient_id.strip())
    except ValueError as e:
        def err():
            yield _sse({"type": "error", "message": str(e)})
        return Response(stream_with_context(err()), mimetype="text/event-stream")

    system_prompt  = build_system_prompt(ctx)
    shap_data      = _data["shap_data"]
    engine         = _data["engine"]

    def generate():
        history_content = user_message
        call_content    = user_message
        sim_result_out  = None

        # ── Simulation detection ─────────────────────────────────────────────
        if sim_engine.is_whatif_question(user_message):
            try:
                changes = sim_engine.parse_whatif_question(user_message, ctx, shap_data)
                if changes == "__best_single__":
                    result = engine.find_best_single_change(patient_id.strip(), shap_data)
                elif changes == "__best_combination__":
                    result = engine.find_best_combination(patient_id.strip(), shap_data)
                else:
                    result = engine.simulate(patient_id.strip(), changes, shap_data)

                sim_block = sim_engine.format_simulation_result(result)
                sim_result_out = result

                history_content = f"{sim_block}\n\nPatient question: {user_message}"
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
                    f"Patient question: {user_message}"
                )

                # Emit the simulation result so the UI can render it
                yield _sse({"type": "sim_result", "result": {
                    k: (v if not isinstance(v, float) else round(v, 4))
                    for k, v in result.items()
                    if not isinstance(v, (np.ndarray, pd.Series))
                }})

            except Exception as exc:
                yield _sse({"type": "sim_warning", "message": f"Simulation error: {exc}"})

        # Build call history (history + augmented user turn + follow-up reminder)
        call_history = history + [{
            "role": "user",
            "content": call_content + _FOLLOW_UP_REMINDER,
        }]

        # ── Stream LLM response ───────────────────────────────────────────────
        token_q: queue.Queue = queue.Queue()
        full_tokens: list[str] = []

        def _call():
            import requests as _req
            import json as _json
            messages = [{"role": "system", "content": system_prompt}] + call_history
            try:
                resp = _req.post(
                    f"{OLLAMA_URL}/api/chat",
                    json={
                        "model": OLLAMA_MODEL,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "stop":           ["\nYou:", "\nUser:", "\nPatient:"],
                            "temperature":    0.1,
                            "repeat_penalty": 1.15,
                            "top_p":          0.9,
                        },
                    },
                    timeout=120,
                    stream=True,
                )
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    chunk = _json.loads(raw_line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        full_tokens.append(token)
                        token_q.put(("token", token))
                    if chunk.get("done"):
                        break
            except Exception as exc:
                token_q.put(("error", str(exc)))
            finally:
                token_q.put(("done", None))

        thread = threading.Thread(target=_call, daemon=True)
        thread.start()

        while True:
            kind, payload = token_q.get()
            if kind == "token":
                yield _sse({"type": "token", "content": payload})
            elif kind == "error":
                yield _sse({"type": "error", "message": payload})
                return
            elif kind == "done":
                full_text = "".join(full_tokens)
                yield _sse({
                    "type":          "metadata",
                    "clean_content": history_content,
                    "response":      full_text,
                })
                yield _sse({"type": "done"})
                return

    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/patient/<patient_id>/simulate", methods=["POST"])
def simulate(patient_id: str):
    """
    Run a specific simulation (non-streaming).
    Request JSON: {"changes": {"FeatureName": value, ...}, "mode": "specific"|"best_single"|"best_combination"}
    """
    body = request.get_json(force=True, silent=True) or {}
    mode = body.get("mode", "specific")
    changes = body.get("changes", {})

    try:
        ctx = _ctx(patient_id.strip())
    except ValueError as e:
        return jsonify({"error": str(e)}), 404

    shap_data = _data["shap_data"]
    engine    = _data["engine"]

    try:
        if mode == "best_single":
            result = engine.find_best_single_change(patient_id.strip(), shap_data)
        elif mode == "best_combination":
            result = engine.find_best_combination(patient_id.strip(), shap_data)
        else:
            if not changes:
                return jsonify({"error": "No changes provided for specific simulation"}), 400
            result = engine.simulate(patient_id.strip(), changes, shap_data)

        # Make JSON-safe
        safe = {}
        for k, v in result.items():
            if isinstance(v, (np.integer,)):   safe[k] = int(v)
            elif isinstance(v, (np.floating,)): safe[k] = float(v)
            elif isinstance(v, np.ndarray):    safe[k] = v.tolist()
            elif isinstance(v, list):
                safe[k] = [
                    {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                     for kk, vv in item.items()} if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                safe[k] = v

        return jsonify(safe)

    except Exception as exc:
        return jsonify({"error": str(exc), "trace": traceback.format_exc()}), 500


@app.route("/api/feature-importance")
def feature_importance():
    """Return feature importance data as JSON."""
    return jsonify(_data.get("feature_importance", []))


@app.route("/api/plots")
def list_plots():
    """Return names of available plot PNG files."""
    plots = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".png")]
    return jsonify(plots)


@app.route("/api/plots/<filename>")
def serve_plot(filename: str):
    """Serve a plot PNG from the outputs directory."""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/api/simulation-meta")
def simulation_meta():
    """Return feature aliases and realistic targets for the simulation UI."""
    return jsonify({
        "feature_aliases":   sim_engine.FEATURE_ALIASES,
        "realistic_targets": {k: v for k, v in sim_engine.REALISTIC_TARGETS.items()},
        "clinical_ranges":   sim_engine.CLINICAL_RANGES,
        "feature_cascades":  sim_engine.FEATURE_CASCADES,
    })


# ---------------------------------------------------------------------------
# SPA catch-all (serve React frontend for any non-API route)
# ---------------------------------------------------------------------------

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path: str):
    if path.startswith("api/"):
        return jsonify({"error": "Not found"}), 404
    dist_index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(dist_index):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return "<h2>Frontend not built yet. Run: cd frontend && npm run build</h2>", 200


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading healthcare data...")
    _load_data()
    n = _data["patient_data"]["PatientID"].nunique()
    print(f"Ready — {n} patients loaded.")
    print("API running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
