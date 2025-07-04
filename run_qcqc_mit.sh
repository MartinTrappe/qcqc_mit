#!/usr/bin/env bash
#
# run_qcqc_mit.sh
#
# Setup and execution script for qcqc_mit.py
#
# Author: Martin-Isbjörn Trappe
# Email: martin.trappe@quantumlah.org
# Date: 2025-07-04
#
# Usage:
#   bash run_qcqc_mit.sh
#
# Creates (if needed) and activates a Python virtual environment `qc-env`,
# installs required dependencies, then runs qcqc_mit.py.
#

set -e

VENV=qc-env

# 1) Create venv if needed
if [ ! -d "$VENV" ]; then
  echo "→ Creating virtual environment '$VENV'…"
  python3 -m venv "$VENV"
else
  echo "→ Virtual environment '$VENV' already exists; skipping."
fi

# 2) Activate
echo "→ Activating '$VENV'…"
. "$VENV/bin/activate"

# 3) Upgrade pip
echo "→ Upgrading pip…"
pip install --upgrade pip

# 4) Helper: install via pip only if import fails
pip_install_if_missing() {
  module="$1"; pkg="$2"
  if ! python3 -c "import ${module}" &> /dev/null; then
    echo "→ Installing ${pkg}…"
    pip install "${pkg}"
  else
    echo "→ ${pkg} already installed, skipping."
  fi
}

# 5) Conditionally install the two deps
pip_install_if_missing qibochem  qibochem
pip_install_if_missing matplotlib matplotlib

# 6) Deactivate
deactivate
echo "✅ Setup complete. To run, do:"
echo "   bash run_qcqc_mit.sh"

# Run the Quantum Chemistry project, e.g. H2_dissociation
./qc-env/bin/python3 qcqc_mit.py

#To completely remove the virtual environment:
#rm -rf qc-env
