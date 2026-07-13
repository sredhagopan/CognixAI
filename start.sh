#!/bin/bash
# CognixAI — Start the healthcare dashboard
# Starts the Flask API server which serves both the REST API
# and the pre-built React frontend.
#
# Usage:
#   ./start.sh              # serve pre-built frontend from frontend/dist/
#   ./start.sh --dev        # run Flask API only (start frontend separately with npm run dev)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check that outputs directory exists
if [ ! -d "outputs" ] || [ ! -f "outputs/predictions_all.csv" ]; then
    echo "ERROR: outputs/ directory missing or incomplete."
    echo "Run the pipeline first: python3 pipeline.py"
    exit 1
fi

# Check that the frontend is built (unless --dev flag)
if [[ "$1" != "--dev" ]] && [ ! -f "frontend/dist/index.html" ]; then
    echo "Frontend not built. Building now..."
    export PATH="$HOME/.local/node/bin:$PATH"
    cd frontend && npm run build && cd ..
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  CognixAI — Healthcare Explainable AI"
echo "═══════════════════════════════════════════════"
echo "  Dashboard: http://localhost:5000"
echo "  API:       http://localhost:5000/api"
echo "  Health:    http://localhost:5000/api/health"
echo "═══════════════════════════════════════════════"
echo ""

# Start Flask
python3 api.py
