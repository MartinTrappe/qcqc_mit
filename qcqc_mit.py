#!/usr/bin/env python3
"""
qcqc_mit.py

driver for Quantum Chemistry via VQE
for dissociation curves of H₂ (and dimers in general)
benchmarking via Qiskit, Qibo, PySCF, & SciPy.

Author: Martin-Isbjörn Trappe
Email: martin.trappe@quantumlah.org
Date: 2025-07-04

Usage:
    bash run_qcqc_mit.sh

See the “BEGIN USER INPUT” / “END USER INPUT” section below to configure:
    • PROJECT name, units, geometry grid, ansatz, optimizer settings, etc.
"""



# ============================
# ===== BEGIN USER INPUT =====
# ============================
PROJECT = "H2_dissociation" # Project name
MIDDLEWARE = 'qibo' # 'qiskit' or 'qibo'
USE_RUNTIME = False  # keep False for local EstimatorV2 runs
USE_QIBOJIT = False  # default: True; set to False for numpy backend
PRINTALL = True # False -- True
MINIMAL_EXAMPLE = True # Minimal example for gauging computing time
THREADS = 10 # os.cpu_count()  # Parallel processes
# Units and geometry sampling
UNITS = 2        # 0: {Ha, Å} -- 1: {eV, Bohr} -- 2: {Ha, Bohr} -- 3: {eV, Å}
xStart = 0.5     # Starting dimer distance (in chosen length units)
xEnd   = 8     # Ending dimer distance
# BBB breakpoints must start with xStart and end with xEnd:
xPoints = [xStart, 0.8, 1.3, 1.39, 1.41, 1.5, 2.0, 2.75, 4.0, 6.0, xEnd]
#xPoints = [xStart,xEnd]
# number of distances in each of the BBB-1 intervals [xPoints[i], xPoints[i+1]] (summming to total number NPTS of bond-length points):
#NPTS_LIST = [10,10,10,10,10,10,10,10,10,10]
NPTS_LIST = [1,1,1,1,1,1,1,1,1,1]
#NPTS_LIST = [3]
# Quantum chemistry and quantum circuit parameters
BASIS = "sto-3g"  # Basis set for PySCF: e.g. sto-3g -- 6-31g {0.03sec/iter} -- 6-31g(d,p) {>3000sec/iter} -- cc-pVDZ -- aug-cc-pVQZ -- aug-cc-pV5Z --
FCIBASIS = "sto-3g"  # Basis set for PySCF: e.g. sto-3g -- 6-31g -- 6-31g(d,p) -- cc-pVDZ -- aug-cc-pVQZ -- aug-cc-pV5Z --
CCSDTBASIS = "sto-3g"  # Basis set for PySCF: e.g. sto-3g -- 6-31g -- 6-31g(d,p) -- cc-pVDZ -- aug-cc-pVQZ -- aug-cc-pV5Z --
ANSATZ = "UCCSD"  # "STD": (layered) hardware-efficient ansatz -- "UCCSD": chemically motivated unitary CC -- "LUCJ": Local Unitary Cluster Jastrow --
ANSATZPARAMS = 1 # For STD: number of layers -- For UCCSD: Number of Trotter steps in operator splitting --
INITAMPLITUDES = "RAND"  # Initial guess: "RAND", "HF", or "MP2" (if UCCSD)
# Classical optimization parameters
OPTIMIZER = "BFGS"   # Classical (scipy) optimizer: CG -- COBYLA -- BFGS -- Nelder-Mead -- Powell -- etc
TOLERANCE = 1e-6 # energy change convergence tolerance for classical optimizer
MAXITER = 1000 # maximum number of iterations for classical optimizer
REPEAT = 3 # select the best result from REPEAT classical optimizer runs
# ==========================
# ===== END USER INPUT =====
# ==========================




# =========================
# ===== BEGIN IMPORTS =====
# =========================


# ---- System ----
import os
# Force all BLAS/MKL/OpenBLAS backends to a single thread to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import json
import shutil
import sys
import time
from datetime import datetime
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ---- Numerics ----
import numpy as np
from scipy.optimize import minimize
from pyscf import gto, scf, fci ,cc, ao2mo

# ---- Plotting ----
import matplotlib.pyplot as plt

# ---- Qibo ----
from qibochem.driver import Molecule  # Quantum chemistry driver: builds molecule & integrals via PySCF
from qibo import set_backend, get_backend, gates, Circuit # Qibo: quantum computing framework
from qibochem.ansatz.ucc import ucc_ansatz, hf_circuit, ucc_circuit  # Unitary Coupled Cluster ansatz builder, etc.
from qibochem.measurement.result import expectation

# ---- OpenFermion ----
from openfermion.circuits import uccsd_singlet_paramsize, uccsd_singlet_generator
from openfermion.chem.molecular_data import spinorb_from_spatial
from openfermion.transforms import jordan_wigner, get_fermion_operator
from openfermion.ops import InteractionOperator

# ---- Qiskit ----
import qiskit
print("\nQiskit version:", qiskit.__version__, "\n")
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

# Prefer Aer EstimatorV2 if present, else SDK's reference EstimatorV2 (ADD)
try:
    from qiskit_aer.primitives import EstimatorV2 as LocalEstimator
except Exception:
    from qiskit.primitives import EstimatorV2 as LocalEstimator  # same V2 interface

if USE_RUNTIME:
    # 1) Import the Runtime client
    from qiskit_ibm_runtime import QiskitRuntimeService

    # 2) Load your IBM creds from the JSON next to this script
    cfg_path = os.path.join(os.path.dirname(__file__), "IBM_apikey.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    # 3) Save (once) so that QiskitRuntimeService() can auto-discover them
    home_cfg = os.path.expanduser("~/.qiskit/qiskit-ibm.json")
    if not os.path.exists(home_cfg):
        QiskitRuntimeService.save_account(
            channel="ibm_quantum_platform",
            token=cfg["apikey"],
            instance=cfg["crn"],
            overwrite=False
        )

    # 4) Instantiate the service (reads from ~/.qiskit/qiskit-ibm.json)
    service = QiskitRuntimeService()

    # 5) Simple check: list available backends
    all_backends = service.backends()
    hardware = [b.name for b in all_backends if not b.configuration().simulator]
    simulators = [b.name for b in all_backends if b.configuration().simulator]

    print("Operational hardware backends:", hardware)
    print("Available simulators:", simulators,"\n")


# =======================
# ===== END IMPORTS =====
# =======================




# ===============================
# ===== BEGIN FILE HANDLING =====
# ===============================

# ── freeze timestamp for parent + children ──
if "QUANTCHEM_TS" not in os.environ:
    os.environ["QUANTCHEM_TS"] = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp = os.environ["QUANTCHEM_TS"]

# ensure our data/ directory exists
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir   = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)
out_dir = os.path.join(data_dir, f"qcqc_mit_{timestamp}")
os.makedirs(out_dir, exist_ok=True)

class Tee(object):
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

# ——— universal Tee (for parent + all children) ———
log_path = os.path.join(out_dir, f"qcqc_{PROJECT}_{timestamp}.log")
# append so that multiple processes write to the same file
log_file = open(log_path, "a")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = sys.stdout

# =============================
# ===== END FILE HANDLING =====
# =============================





# ============================================
# ===== BEGIN POST-PROCESSING USER INPUT =====
# ============================================

if MIDDLEWARE == 'qiskit':
    USE_QIBOJIT = False
    print("Test local qiskit simulator...")
    # 1) Build your circuit (e.g. Bell)
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    # 2) Add measurements
    qc.measure_all()
    # 3) Pick the local simulator backend
    sim = AerSimulator()
    # 4) Transpile to match the simulator’s expected instruction set
    qc2 = transpile(qc, sim)
    # 5) Run and fetch results
    job = sim.run(qc2, shots=1024)
    result = job.result()
    print("Counts:", result.get_counts())
    print("...local qiskit simulator works\n")


if USE_QIBOJIT:
    set_backend("qibojit") # for CPU
    #set_backend("qibojit", platform="gpu")  # if using CuPy/CUDA
else:
    set_backend("numpy")
backend = get_backend()
print(f"\n → Qibo backend = {backend.name}, platform = {backend.platform}\n")
logging.getLogger("qibo").setLevel(logging.DEBUG)
logging.getLogger('qiskit').setLevel(logging.WARNING)


if ANSATZ == "STD":
    INITAMPLITUDES = "RAND"
if ANSATZ == "LUCJ" and INITAMPLITUDES == "MP2":
    INITAMPLITUDES = "HF"


# --- Unit conversion setup ---
if UNITS == 0:
    length_unit = "Å";   energy_unit = "Ha"
    convert_distance = 1                 # input is already in Å
    convert_energy = 1                   # HF & VQE energies in Hartree
elif UNITS == 1:
    length_unit = "Bohr"; energy_unit = "eV"
    convert_distance = 1/0.529177210544  # Å → Bohr
    convert_energy = 27.2114079527       # Ha → eV
elif UNITS == 2:
    length_unit = "Bohr"; energy_unit = "Ha"
    convert_distance = 1/0.529177210544  # Å → Bohr
    convert_energy = 1                   # keep Hartree
elif UNITS == 3:
    length_unit = "Å"; energy_unit = "eV"
    convert_distance = 1                 # keep Å
    convert_energy = 27.2114079527       # Ha → eV

if MINIMAL_EXAMPLE == True:
    THREADS = 1

CircuitSavedQ = False

# ==========================================
# ===== END POST-PROCESSING USER INPUT =====
# ==========================================





# ==================================
# ===== BEGIN HELPER FUNCTIONS =====
# ==================================

def SleepForever():
    try:
        print("Sleeping forever. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)  # sleep 1 second per loop
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")


def _qop_to_sparse_pauli_op(qop, n_qubits, reverse_qubit_order=True):
    """OpenFermion QubitOperator → Qiskit SparsePauliOp."""
    # Handle the case of a zero operator (empty terms)
    if not qop.terms:
        return SparsePauliOp("I" * n_qubits, 0)

    paulis, coeffs = [], []
    for term, coeff in qop.terms.items():
        if abs(np.imag(coeff)) > 1e-9: continue
        p_str = ['I'] * n_qubits
        for qi, op in term:
            p_str[qi] = op

        if reverse_qubit_order:
            paulis.append("".join(p_str[::-1]))
        else:
            paulis.append("".join(p_str))
        coeffs.append(np.real(coeff))

    return SparsePauliOp.from_list(list(zip(paulis, coeffs)))


def _uccsd_singlet_generator_safe(theta_vec, n_so, n_elec):
    """
    OpenFermion changed arg order across versions; try both.
    Returns a FermionOperator representing (T - T†) (anti-Hermitian up to factor i).
    """
    try:
        # common signature
        return uccsd_singlet_generator(n_so, n_elec, theta_vec, anti_hermitian=True)
    except TypeError:
        # older/newer alt signature
        return uccsd_singlet_generator(theta_vec, n_so, n_elec, anti_hermitian=True)

# ================================
# ===== END HELPER FUNCTIONS =====
# ================================





# ==========================================
# ===== BEGIN MAIN COMPUATIONAL MODULE =====
# ==========================================

def compute_energy(raw_d, init_params=None):
    """
    Compute SCF, VQE, FCI, and CCSD(T) energies for a molecule at a given distance.
    """
    global CircuitSavedQ

    distance = raw_d / convert_distance

    # --- 1. Classical Chemistry (Common to both backends) ---
    if PRINTALL: print(" --- 1a) Build molecule with qibochem and run mean-field RHF ---")
    mol = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, distance))], basis=BASIS)
    mol.run_pyscf()
    scf_energy = mol.e_hf
    print("HF done.")

    if PRINTALL: print(" --- 1b) Run FCI benchmark ---")
    ps_mol = gto.M(atom=[('H', (0, 0, 0)), ('H', (0, 0, distance))], basis=FCIBASIS, verbose=0, output=None)
    ps_mf = scf.RHF(ps_mol).run(verbose=0)
    fci_energy, _ = fci.FCI(ps_mf).kernel()
    if PRINTALL: print("FCI done.")

    if PRINTALL: print(" --- 1c) Run CCSD(T) benchmark ---")
    mycc = cc.CCSD(ps_mf).run(verbose=0)
    ccsd_t_energy = mycc.e_tot + mycc.ccsd_t()
    if PRINTALL: print("CCSD(T) done.")


    # --- 2. Backend-Specific VQE Setup ---
    vqe_objective = None
    n_params = 0

    if MIDDLEWARE == 'qibo':
        if PRINTALL: print(" --- 2a) Constructing Qibo Hamiltonian and Ansatz ---")
        hamiltonian = mol.hamiltonian()

        # Build Ansatz
        if ANSATZ == "UCCSD":
            circuit = ucc_ansatz(mol, trotter_steps=ANSATZPARAMS, include_hf=True, use_mp2_guess=(INITAMPLITUDES == "MP2"))
        else: # Add other Qibo-supported ansatz here if needed
            raise ValueError(f"Unsupported ANSATZ '{ANSATZ}' for Qibo backend")

        n_params = len(circuit.get_parameters())

        # Define Objective Function
        def objective_qibo(params):
            circuit.set_parameters(params)
            return expectation(circuit, hamiltonian)
        vqe_objective = objective_qibo

        # Handle MP2 initial guess specifically for Qibo
        if INITAMPLITUDES == "MP2" and ANSATZ == "UCCSD":
            init_params = np.asarray(circuit.get_parameters()).flatten()

        if PRINTALL and not CircuitSavedQ:
            CircuitSavedQ = True
            print(f"PRINTALL (compute_energy): Distance={distance:.4f} Bohr, Basis={BASIS}")
            param_gates = {gates.RY, gates.RZ, gates.RX}

            # ── count single- and two-qubit gates ──
            ops = circuit.queue      # the list of Gate objects in your circuit
            n_single = sum(1 for gate in ops if len(gate.qubits) == 1)
            n_two    = sum(1 for gate in ops if len(gate.qubits) == 2)

            param_indices = [i for i, op in enumerate(ops) if type(op) in param_gates]
            nqubits = mol.nso
            print(f"       Number of qubits             = {nqubits}")
            print(f"       Number of circuit gates      = {n_single} single-qubit, {n_two} two-qubit")
            print(f"       Number of circuit parameters = {n_params}")
            print(circuit.summary())
            sys.stdout.flush()

    elif MIDDLEWARE == 'qiskit':
        if PRINTALL: print(" --- 2a) Constructing Qiskit Hamiltonian and Ansatz ---")

        # --- Build Hamiltonian directly from qibochem integrals ---
        n_so = mol.nso
        n_elec = mol.nelec

        one_body_integrals = mol.oei
        two_body_integrals = mol.tei
        one_body_spin_ints, two_body_spin_ints = spinorb_from_spatial(one_body_integrals, two_body_integrals)

        hamiltonian_op = InteractionOperator(0, one_body_spin_ints, 0.5 * two_body_spin_ints)
        fermion_hamiltonian = get_fermion_operator(hamiltonian_op)
        qubit_hamiltonian_of = jordan_wigner(fermion_hamiltonian)

        pauli_list = []
        for op, coeff in qubit_hamiltonian_of.terms.items():
            if abs(coeff.imag) > 1e-9: continue
            pauli_str = ['I'] * n_so
            for qubit_idx, pauli_op in op:
                pauli_str[qubit_idx] = pauli_op
            pauli_list.append(("".join(reversed(pauli_str)), coeff.real))

        # 1. Define the electronic Hamiltonian (quantum part)
        H_electronic = SparsePauliOp.from_list(pauli_list)
        # 2. Store the nuclear repulsion (classical part) separately
        nuclear_repulsion = mol.e_nuc

        # --- Build Ansatz (UCCSD) ---
        n_params = uccsd_singlet_paramsize(n_so, n_elec)

        # --- Define Objective Function ---
        estimator = LocalEstimator()
        def objective_qiskit(theta_vec):
            # Create the UCCSD circuit for the given theta parameters
            gen_ferm = _uccsd_singlet_generator_safe(theta_vec, n_so, n_elec)
            gen_ferm = 1j * gen_ferm  # Make Hermitian
            gen_qubit = jordan_wigner(gen_ferm)
            H_theta = _qop_to_sparse_pauli_op(gen_qubit, n_so, reverse_qubit_order=True)

            # Build the full circuit: HF state + UCCSD evolution
            qc = QuantumCircuit(n_so)
            # Occupy the lowest energy orbitals for the HF state
            for i in range(n_elec):
                qc.x(i)
            qc.append(PauliEvolutionGate(H_theta, time=1.0), qc.qubits)

            # Transpile for the simulator
            transpiled_qc = transpile(qc, backend=sim)

            # Calculate expectation value of the electronic part
            pub = (transpiled_qc, H_electronic, []) # Use the electronic Hamiltonian
            result = estimator.run([pub]).result()

            # 3. Add the classical nuclear energy to the final quantum result
            return result[0].data.evs + nuclear_repulsion

        vqe_objective = objective_qiskit


    # --- 3. Common VQE Optimization Loop ---
    if PRINTALL: print(" --- 3) Preparing for classical optimization ---")

    # Define initial parameters
    if init_params is None:
        if INITAMPLITUDES == "HF":
            init_params = np.zeros(n_params)
        else: # "RAND"
            init_params = np.random.uniform(0, 2 * np.pi, n_params)

    # Define callback for optimizer
    callback = None
    if PRINTALL and abs(raw_d - xStart) < 1e-6:
        iter_count = {'n': 0}
        def scipy_cb(xk):
            iter_count['n'] += 1
            t0 = time.perf_counter()
            ek = vqe_objective(xk)
            dt = time.perf_counter() - t0
            print(f"Iter {iter_count['n']:2d}: energy = {ek:.8f} Ha, eval_time = {dt:.3f} s")
            sys.stdout.flush()
        callback = scipy_cb

    # Run classical optimization
    best_result = None
    for r in range(REPEAT):
        current_init_params = init_params
        if r > 0 and INITAMPLITUDES != "HF":
            current_init_params = np.random.uniform(0, 2 * np.pi, n_params)

        result = minimize(
            vqe_objective,
            current_init_params,
            method=OPTIMIZER,
            tol=TOLERANCE,
            options={"maxiter": MAXITER, "disp": True},
            callback=callback
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    best_params = best_result.x
    best_energy = best_result.fun

    if PRINTALL: print("Energy computed.")

    return scf_energy, best_energy, fci_energy, ccsd_t_energy, best_params

# ========================================
# ===== END MAIN COMPUATIONAL MODULE =====
# ========================================





# ==============================
# ===== BEGIN MAIN PROGRAM =====
# ==============================

def main():
    # Record wall-clock time for performance metrics
    start = time.perf_counter()
    np.random.seed(0)  # reproducible random guesses

    # --- Logic for Distances ---
    if MINIMAL_EXAMPLE:
        print("\n--- Running MINIMAL EXAMPLE (single distance calculation) ---")
        distances = np.array([xStart])
        NPTS = 1
    else:
        print("\n--- Running Full Dissociation Curve Calculation ---")
        # Generate bond-length grid
        segments = []
        for (x0, x1), count in zip(zip(xPoints[:-1], xPoints[1:]), NPTS_LIST):
            segments.append(np.linspace(x0, x1, count, endpoint=False))
        segments[-1] = np.linspace(xPoints[-2], xPoints[-1], NPTS_LIST[-1], endpoint=True)
        distances = np.concatenate(segments)
        NPTS = distances.size

    print(f"\n ^^^^^^^^ Starting Calculation ({NPTS} point(s)) ^^^^^^^^\n")

    # --- VQE Execution (Serial or Parallel) ---
    results = []
    if THREADS == 1:
        last_params = None
        for i, d in enumerate(distances):
            print(f"\n--- Calculating point {i+1}/{NPTS} at distance {d:.4f} ---")
            scf_e, vqe_e, fci_e, ccsd_t_e, last_params = compute_energy(d, init_params=last_params)
            # Append a single tuple to the results list
            results.append((d, scf_e, vqe_e, fci_e, ccsd_t_e))
    else: # Parallel processing
        with ProcessPoolExecutor(max_workers=THREADS) as executor:
            future_to_dist = {executor.submit(compute_energy, d): d for d in distances}
            temp_results = {}
            for i, future in enumerate(as_completed(future_to_dist)):
                dist = future_to_dist[future]
                print(f"\n--- Completed point {i+1}/{NPTS} (distance {dist:.4f}) ---")
                scf_e, vqe_e, fci_e, ccsd_t_e, _ = future.result()
                temp_results[dist] = (dist, scf_e, vqe_e, fci_e, ccsd_t_e)
            # Sort results by distance to ensure correct plot order
            results = [temp_results[d] for d in sorted(temp_results)]

    # --- UNIFIED Logging and Unpacking (runs once) ---
    log_header = f"#{'distance':>10} {'scf_energy':>16} {'vqe_energy':>16} {'fci_energy':>16} {'ccsdt_energy':>16}"
    logging.info(log_header)
    for d, scf_e, vqe_e, fci_e, ccsdt_e in results:
        logging.info(f"{d:11.4f} {scf_e*convert_energy:16.8f} {vqe_e*convert_energy:16.8f} {fci_e*convert_energy:16.8f} {ccsdt_e*convert_energy:16.8f}")

    # === Total runtime ===
    elapsed = time.perf_counter() - start
    print(f"\nTotal wall-clock time: {elapsed:.2f} seconds")

    # --- Plotting (only if not a minimal example) ---
    if not MINIMAL_EXAMPLE:
        # Unpack the results for plotting
        dist_plot = [res[0] for res in results]
        scf_plot = [res[1] * convert_energy for res in results]
        vqe_plot = [res[2] * convert_energy for res in results]
        fci_plot = [res[3] * convert_energy for res in results]
        ccsdt_plot = [res[4] * convert_energy for res in results]

        plt.plot(dist_plot, scf_plot, 'o', color='blue', label='HF')
        plt.plot(dist_plot, vqe_plot, '--', color='orange', label=f"VQE [{BASIS}]")
        plt.plot(dist_plot, fci_plot, '-', color='lightgreen', label=f"FCI [{FCIBASIS}]")
        plt.plot(dist_plot, ccsdt_plot, ':', color='black', label=f"CCSD(T) [{CCSDTBASIS}]")

        # Annotate minimum VQE energy point
        idx_min = np.argmin(vqe_plot)
        min_d, min_e = dist_plot[idx_min], vqe_plot[idx_min]
        plt.scatter(min_d, min_e, marker='*', color='red', s=80, zorder=5)
        plt.annotate(
            f"VQE: {{ {min_e:.4f} {energy_unit} @ {min_d:.4f} {length_unit} }}",
            xy=(min_d, min_e),
            xytext=(min_d + 0.1*(dist_plot[-1]-dist_plot[0]), min_e + 0.05*(max(vqe_plot)-min_e)),
            color='black',
            arrowprops=dict(arrowstyle='->',color='black')
        )

        plt.legend()
        plt.xlabel(f"distance [{length_unit}]")
        plt.ylabel(f"energy [{energy_unit}]")
        plt.title(f"{PROJECT}")
        plt.grid(True)
        plt.tight_layout()

        # Save figure with timestamp and parameter summary
        ANSATZstring = f"{ANSATZ}({ANSATZPARAMS})"
        plt.savefig(os.path.join(out_dir, f"qcqc_{PROJECT}_{timestamp}_[{xStart}-{xEnd}]({NPTS})_{ANSATZstring}_{INITAMPLITUDES}_{BASIS}_FCI={FCIBASIS}_CCSDT={CCSDTBASIS}_{OPTIMIZER}_{elapsed:.0f}sec.pdf"))
        plt.show()


# ============================
# ===== END MAIN PROGRAM =====
# ============================




if __name__ == "__main__":
    # 1) Multiprocessing start method
    mp.set_start_method("spawn", force=True)

    # 2) Configure logger to write .dat and stream to the Tee’d stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(out_dir, f"qcqc_{PROJECT}_{timestamp}.dat"), mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 3) Self-backup and its print() now goes through the Tee
    if mp.current_process().name == "MainProcess":
        script_path = os.path.realpath(__file__)
        backup_name = os.path.join(out_dir, f"qcqc_{PROJECT}_{timestamp}.py")
        shutil.copy(script_path, backup_name)
        print(f"Created backup of script as {backup_name}")

    # 4) Finally run the main computation
    main()
