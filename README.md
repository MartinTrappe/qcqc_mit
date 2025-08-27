# qcqc_mit

**Quantum Computing for Quantum Chemistry**

Multi-process driver for Hartree–Fock (HF), VQE, FCI and CCSD(T) dissociation curves using VQE via Qiskit & Qibo.

---

## Overview

`qcqc_mit` computes dissociation curves for small molecules (e.g. H₂) by:

1. Running mean-field RHF (Restricted Hartree–Fock) via PySCF
2. Building FCI benchmark in chosen basis
3. Optionally CCSD(T) reference
4. Mapping to qubits & constructing a VQE ansatz (hardware-efficient or UCCSD)
5. Running VQE via Qiskit/Qibo with SciPy optimizers
6. Collecting, logging and plotting all energies

All output is saved in a timestamped `data/` folder, along with a backup of the script.

---

## Author

**Martin-Isbjörn Trappe**

Email: martin.trappe@quantumlah.org

4 July 2025

---

## Features

- **Parallel execution** (sampling the distances of the dissociation curve in parallel)
- **Classical optimizers**: BFGS, CG, COBYLA, Nelder-Mead, etc.
- **Automatic backup** of script and logs
- **Publication-ready plots** of HF, VQE, FCI & CCSD(T) curves

---

## Prerequisites

- **Python 3.7+**
- **bash** (for `run_qcqc_mit.sh`)
- **PySCF**, **numpy**, **matplotlib**, **pandas**, **qiskit**, **qibochem**, **qibo**, **scipy**, and more (will be installed automatically if missing)

---

## Installation

Clone and enter repo:

```bash
git clone https://github.com/MartinTrappe/qcqc_mit.git
cd qcqc_mit
```

## File Structure

```
.
├── qcqc_mit.py         # Main QC driver
├── run_qcqc_mit.sh     # Setup & run wrapper
├── data/               # Logs, backups & plots (auto-created)
├── README.md           # This file
├── LICENSE             # (if present)
└── .gitignore          # e.g. qc-env/, data/, etc.
```

## Usage

**Edit** the `# === USER INPUT ===` block in `qcqc_mit.py` to set your parameters:

   - Middleware backend (qiskit/qibo)
   - VQE BASIS set
   - FCIBASIS set
   - CCSDTBASIS set
   - quantum computing ANSATZ
   - classical OPTIMIZER
   - Number of THREADS
   - Control parameters, etc.

**Run** the setup and script:
   ```bash
   ./run_qcqc_mit.sh
   ```

**Inspect** results in the newly created `data/` folder.


