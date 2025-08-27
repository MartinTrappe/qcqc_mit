#!/usr/bin/env bash
#
# run_qcqc_mit.sh
#
# Setup and execution script for qcqc_mit.py
#
# Author: Martin-Isbjörn Trappe
# Email: martin.trappe@quantumlah.org
# Date: 2025-07-07
#
# Usage:
#   bash run_qcqc_mit.sh
#
# This script:
#   - Creates (if needed) a *clean* Python virtual environment `qc-env` (without system packages)
#   - Installs required Python packages via pip
#   - Executes the quantum chemistry simulation script `qcqc_mit.py`
#
# This version avoids system Python package interference and is suitable for publication, reproducibility,
# and portability across Linux distributions with Python ≥ 3.10.

set -e

VENV=qc-env
PYTHON=python3

# 1) Create a clean virtual environment if it doesn't exist
if [ ! -d "$VENV" ]; then
  echo "→ Creating clean virtual environment '$VENV'…"
  "$PYTHON" -m venv "$VENV"
else
  echo "→ Virtual environment '$VENV' already exists; skipping."
fi

# 2) Activate the environment
echo "→ Activating '$VENV'…"
source "$VENV/bin/activate"

# 3) Upgrade pip inside the venv
echo "→ Updating pip…"
pip install --quiet --upgrade pip

# 4) Install required packages via pip

# ——— Conditionally install Qiskit[all] ———
echo "→ Updating Qiskit…"
if ! python -c "import qiskit" &> /dev/null; then
  echo "→ Qiskit not found; installing Qiskit[all]…"
  pip install --quiet "qiskit[all]"
else
  echo "→ Qiskit already installed; skipping."
fi


# These packages will not conflict with system-level packages
echo "→ Updating required Python packages…"
pip install --quiet matplotlib qibochem qibojit qiskit_ibm_runtime qiskit-aer openfermion

# 5) Run the Quantum Chemistry project, e.g. H2_dissociation
echo "→ Running quantum chemistry simulation: qcqc_mit.py"
python qcqc_mit.py

# 6) Deactivate the environment (optional here, but clean practice)
deactivate

echo "✅ Simulation complete. Environment remains at './$VENV'."


#To completely remove the virtual environment:
#rm -rf qc-env
