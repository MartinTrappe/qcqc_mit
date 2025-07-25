#!/usr/bin/env python3
"""
qcqc_mit.py

Quantum Chemistry + VQE dissociation curve driver
for H₂ (and similar) benchmarking via PySCF, Qibochemical, Qibo & SciPy.

Author: Martin-Isbjörn Trappe
Email: martin.trappe@quantumlah.org
Date: 2025-07-04

Usage:
    bash run_qcqc_mit.sh

Dependencies:
    - qibochem
    - qibo
    - pyscf
    - scipy
    - numpy
    - matplotlib
    - pandas

See the “BEGIN USER INPUT” / “END USER INPUT” section below to configure:
    • PROJECT name, units, geometry grid, ansatz, optimizer settings, etc.
"""



# ============================
# ===== BEGIN USER INPUT =====
# ============================
PROJECT = "H2_dissociation" # Project name
USE_QIBOJIT = False  # default: True; set to False for numpy backend
DEBUG = True # False -- True
MINIMAL_EXAMPLE = True # Minimal example for gauing computing time
THREADS = 1 # os.cpu_count()  # Parallel processes
# Units and geometry sampling
UNITS = 2        # 0: {Ha, Å} -- 1: {eV, Bohr} -- 2: {Ha, Bohr} -- 3: {eV, Å}
xStart = 0.5     # Starting H–H distance (in chosen length units)
xEnd   = 8     # Ending   H–H distance
# breakpoints must start with xStart and end with xEnd:
xPoints = [xStart, 0.8, 1.3, 1.39, 1.41, 1.5, 2.0, 2.75, 4.0, 6.0, xEnd]
# number of distances in each interval [xPoints[i], xPoints[i+1]] (summming to total number NPTS of bond-length points):
#NPTS_LIST = [10,10,10,10,10,10,10,10,10,10]
NPTS_LIST = [1,1,1,1,1,1,1,1,1,1]
# Quantum chemistry and quantum circuit parameters
BASIS = "sto-3g"  # Basis set for PySCF: e.g. sto-3g -- 6-31g {0.03sec/iter} -- 6-31g(d,p) {>3000sec/iter} -- cc-pVDZ -- aug-cc-pVQZ -- aug-cc-pV5Z --
FCIBASIS = "sto-3g"  # Basis set for PySCF: e.g. sto-3g -- 6-31g -- 6-31g(d,p) -- cc-pVDZ -- aug-cc-pVQZ -- aug-cc-pV5Z --
CCSDTBASIS = "sto-3g"  # Basis set for PySCF: e.g. sto-3g -- 6-31g -- 6-31g(d,p) -- cc-pVDZ -- aug-cc-pVQZ -- aug-cc-pV5Z --
ANSATZ = "STD"  # "STD": (layered) hardware-efficient ansatz -- "UCCSD": chemically motivated unitary CC -- "LUCJ": Local Unitary Cluster Jastrow --
ANSATZPARAMS = 1 # For STD: number of layers -- For UCCSD: Number of Trotter steps in operator splitting --
INITAMPLITUDES = "HF"  # Initial guess: "RAND", "HF", or "MP2" (if UCCSD)
# Classical optimization parameters
OPTIMIZER = "BFGS"   # Classical scipy optimizer: CG -- COBYLA -- BFGS -- Nelder-Mead -- Powell --
TOLERANCE = 1e-2 # energy change convergence tolerance for classical optimizer
MAXITER = 1000 # maximum number of iterations for classical optimizer
REPEAT = 10 # select the best result from REPEAT classical optimizer runs
# ==========================
# ===== END USER INPUT =====
# ==========================



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
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp

import numpy as np
from scipy.optimize import minimize
from pyscf import lib, gto, scf, fci ,cc
from qibochem.driver import Molecule  # Quantum chemistry driver: builds molecule & integrals via PySCF
from qibo import set_backend, get_backend, gates, Circuit # Qibo: quantum computing framework
from qibo.models import VQE # Variational Quantum Eigensolver model
from qibochem.ansatz.ucc import ucc_ansatz, hf_circuit, ucc_circuit  # Unitary Coupled Cluster ansatz builder, etc.
from qibochem.measurement.result import expectation
from qibo.ui import plot_circuit


import qiskit

print("\nQiskit version:", qiskit.__version__, "\n")

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

print("Local qiskit simulator:")

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

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
job    = sim.run(qc2, shots=1024)
result = job.result()
print("Counts:", result.get_counts())




if USE_QIBOJIT:
    set_backend("qibojit") # for CPU
    #set_backend("qibojit", platform="gpu")  # if using CuPy/CUDA
else:
    set_backend("numpy")
from qibo import get_backend
backend = get_backend()
print(f"\n → Qibo backend = {backend.name}, platform = {backend.platform}\n")
import logging
logging.getLogger("qibo").setLevel(logging.DEBUG)


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


def SleepForever():
    try:
        print("Sleeping forever. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)  # sleep 1 second per loop
    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")


# ── freeze timestamp for parent + children ──
if "QUANTCHEM_TS" not in os.environ:
    os.environ["QUANTCHEM_TS"] = datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp = os.environ["QUANTCHEM_TS"]


# ensure our data/ directory exists
script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir   = os.path.join(script_dir, "data")
os.makedirs(data_dir, exist_ok=True)


# ——— universal Tee (for parent + all children) ———
log_path = os.path.join(data_dir, f"qcqc_{PROJECT}_{timestamp}.log")
# append so that multiple processes write to the same file
log_file = open(log_path, "a")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = sys.stdout

# --- Post-Process user input ---

if MINIMAL_EXAMPLE:
    print("start MINIMAL EXAMPLE for gauging computing time")

    # --- Molecule setup: H₂ with accurate basis, e.g. sto-3g -- 6-31g -- 6-31g(d,p) -- cc-pVDZ -- aug-cc-pVQZ -- aug-cc-pV5Z --
    DISTANCE = 0.7
    h2 = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, DISTANCE))], basis=BASIS)
    h2.run_pyscf()
    hamiltonian = h2.hamiltonian()

    # --- UCC ansatz: HF reference + full excitations
    print(f"→ Spin orbitals: {h2.nso}, electrons: {h2.nelec}")
    circuit = hf_circuit(h2.nso, h2.nelec)
    circuit += ucc_circuit(h2.nso, list(range(h2.nso)))

    # --- VQE setup
    vqe = VQE(circuit, hamiltonian)
    nparams = len(circuit.get_parameters())
    initial_parameters = np.random.uniform(0.0, 2*np.pi, nparams)

    print(f"→ Circuit has {nparams} parameters")

    # --- Callback for live progress updates
    def callback(xk):
        t0 = time.perf_counter()
        circuit.set_parameters(xk)
        result = circuit()
        state = result.state()  # ← Extract raw statevector
        energy = hamiltonian.expectation(state)
        t1 = time.perf_counter()
        print(f"→ Iter energy = {energy:.8f} Ha, time = {t1 - t0:.3f}s")
        sys.stdout.flush()

    # --- Minimize
    print("→ Starting VQE optimization…")
    t0 = time.perf_counter()
    energy, params, extra = vqe.minimize(
        initial_parameters,
        method="BFGS",
        options={"maxiter": 1000, "disp": True},
        callback=callback
    )
    t1 = time.perf_counter()

    # --- Results
    print(f"\n✅ VQE ground state energy at DISTANCE={DISTANCE} in {BASIS} basis: {energy:.10f} Ha")
    print(f"✅ Optimization time: {t1 - t0:.2f} seconds")

    print(" end MINIMAL EXAMPLE for gauging computing time\n")



# def lucj_ansatz(n_qubits, n_layers=1):
#     """
#     Local Unitary Cluster Jastrow (LUCJ) ansatz:
#     Alternating layers of parameterized RZZ entanglers and RY rotations.
#     """
#     from qibo import gates, Circuit
#     circ = Circuit(n_qubits)
#
#     for l in range(n_layers):
#         # --- Jastrow-like ZZ entanglers ---
#         for i in range(n_qubits - 1):
#             theta = 0.0
#             circ.add(gates.RZZ(i, i + 1, theta=theta))
#
#         # --- Local RY rotations ---
#         for i in range(n_qubits):
#             theta = 0.0
#             circ.add(gates.RY(i, theta=theta))
#
#     return circ

# def lucj_ansatz(n_qubits, n_layers=1):
#     from qibo import gates, Circuit
#     from qibochem.ansatz.ucc import hf_circuit
#
#     circ = Circuit(n_qubits)
#
#     if INITAMPLITUDES == "HF":
#         circ += hf_circuit(n_qubits, n_electrons=n_qubits // 2)
#
#     for l in range(n_layers):
#         # --- Choose entangler type ---
#         if l % 3 == 0:
#             entangler = gates.RZZ
#             name = "zz"
#         elif l % 3 == 1:
#             entangler = gates.RXX
#             name = "xx"
#         else:
#             entangler = gates.RYY
#             name = "yy"
#
#         # All-to-all entanglers
#         for i in range(n_qubits):
#             for j in range(i+1, n_qubits):
#                 circ.add(entangler(i, j, theta=0.0))
#
#         # Single-qubit RY rotations
#         for i in range(n_qubits):
#             circ.add(gates.RY(i, theta=0.0))
#
#     return circ
from qibo import gates, Circuit
from qibochem.ansatz.ucc import hf_circuit

def enhanced_lucj_ansatz(n_qubits, n_layers=2, include_hf=False, use_parameter_sharing=False):
    """
    Enhanced LUCJ ansatz:
    - Alternating entanglers (RZZ, RXX, RYY)
    - All-to-all connectivity per layer
    - Optional HF reference state
    - Optional parameter sharing across layers
    - Placeholder SWAP layers for symmetry (fixed for now)

    Args:
        n_qubits (int): number of qubits
        n_layers (int): number of LUCJ layers
        include_hf (bool): whether to prepend Hartree–Fock reference circuit
        use_parameter_sharing (bool): whether to share parameters across layers

    Returns:
        qibo.Circuit: parameterized LUCJ circuit
    """
    circ = Circuit(n_qubits)

    if include_hf:
        circ += hf_circuit(n_qubits, nelectrons=n_qubits // 2)

    # Entangler types alternating
    entangler_sequence = [gates.RZZ, gates.RXX, gates.RYY]

    # Optional: parameter table for sharing
    shared_params = {}

    for l in range(n_layers):
        entangler = entangler_sequence[l % len(entangler_sequence)]

        # All-to-all entanglers
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                if use_parameter_sharing:
                    key = (entangler.__name__, i, j)
                    if key not in shared_params:
                        shared_params[key] = 0.0
                    theta = shared_params[key]
                else:
                    theta = 0.0
                circ.add(entangler(i, j, theta=theta))

        # Optional symmetry-preserving SWAPs (placeholder, unparameterized)
        if n_qubits >= 4:
            for i in range(0, n_qubits - 1, 2):
                circ.add(gates.SWAP(i, i+1))

        # RY single-qubit layer
        for i in range(n_qubits):
            circ.add(gates.RY(i, theta=0.0))

    return circ



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

CircuitSavedQ = False

def compute_energy(raw_d, init_params=None):
    """
    Compute SCF (Hartree–Fock), VQE (Variation Quantum Eigensolver), and FCI (Full Configuration Interaction) energies for molecule at a given bond length.

    Physics/Chemistry:
      1) Build molecule geometry, run Restricted Hartree–Fock (RHF) via PySCF to get mean-field reference.
      2) Extract molecular Hamiltonian in second-quantized form.

    Quantum Computing (based on Qibo):
      3) Map fermionic Hamiltonian to qubit operator (Jordan–Wigner by default).
      4) Construct ansatz circuit (hardware-efficient or UCCSD).
      5) Run VQE: classical outer loop optimizing parameters to minimize expectation value.
    """

    global CircuitSavedQ

    distance = raw_d/convert_distance

    if DEBUG:
        print(" --- 1a) Build molecule with qibochem and run mean-field RHF ---")
        sys.stdout.flush()
    mol = Molecule([('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, distance))], basis=BASIS)
    mol.run_pyscf()
    scf_energy = mol.e_hf  # Hartree–Fock energy in atomic units (Hartree)
    print("HF done.")
    sys.stdout.flush()

    if DEBUG:
        print(" --- 1b) Build the same molecule in FCI for quasi-exact benchmark (of the chosen FCI basis set) ---")
        sys.stdout.flush()
    ps_mol = gto.M(atom=[('H',(0,0,0)), ('H',(0,0,distance))], basis=FCIBASIS, verbose=0, output=None)
    ps_mf = scf.RHF(ps_mol).run()
    ps_mf.verbose = 0
    # Create an FCI solver from the RHF object, kernel() returns (energy, coeffs)
    cis = fci.FCI(ps_mf)
    fci_energy, _ = cis.kernel() # FCI is exact (for the chosen basis) and gives the true value in the basis set limit
    if DEBUG:
        print("FCI done.")
        sys.stdout.flush()

    if DEBUG:
        print(" --- 1c) Run CCSD(T) on the same molecule ---")
        sys.stdout.flush()
    ps_mol = gto.M(atom=[('H',(0,0,0)), ('H',(0,0,distance))], basis=CCSDTBASIS, verbose=0, output=None)
    mycc = cc.CCSD(ps_mf).run()
    ccsd_t_energy = mycc.e_tot + mycc.ccsd_t() # for 2-electron systems like H2, the triple excitation is necessarily zero
    if DEBUG:
        print("CCSD(T) done.")
        sys.stdout.flush()

    if DEBUG:
        print(" --- 2) Construct fermionic Hamiltonian and map to qubits ---")
        sys.stdout.flush()
    H = mol.hamiltonian()
    nqubits = H.nqubits


    if DEBUG:
        print(" --- 3) Build variational ansatz circuit ---")
        sys.stdout.flush()
    if ANSATZ == "STD":
        # (Layered) Hardware-efficient ansatz: ANSATZPARAMS layers of single-qubit RY rotations + chain of CNOTs
        circ = Circuit(nqubits)
        for _ in range(ANSATZPARAMS):
            for i in range(nqubits):
                circ.add(gates.RY(i, theta=0.0))
            for i in range(nqubits - 1):
                circ.add(gates.CNOT(i, i+1))
    elif ANSATZ == "UCCSD":
        # Chemically motivated UCCSD: singles & doubles excitations, trotterized
        circ = ucc_ansatz(
            mol,
            trotter_steps=ANSATZPARAMS,
            ferm_qubit_map="jw",
            include_hf=True,
            use_mp2_guess=(INITAMPLITUDES == "MP2")
        )
    elif ANSATZ == "LUCJ":
        circ = enhanced_lucj_ansatz(nqubits, n_layers=ANSATZPARAMS, include_hf=False, use_parameter_sharing=True)
    else:
        raise ValueError(f"Unknown ansatz: {ANSATZ}")

    if DEBUG:
        print(" --- 4) Initialize VQE model with Qibo ---")
        sys.stdout.flush()
    vqe = VQE(circ, H)
    # Prepare initial parameter guess
    nparams = len(circ.get_parameters())
    if DEBUG and not CircuitSavedQ:
        CircuitSavedQ = True
        print(f"DEBUG (compute_energy): Distance={distance:.4f} Bohr, Basis={BASIS}")
        # ── count basis functions for the QC basis ──
        qc_mol = gto.M(
            atom=[('H', (0,0,0)), ('H', (0,0,distance))],
            basis=BASIS,
            verbose=0, output=None
        )
        n_spatial = qc_mol.nao_nr()               # number of spatial AOs
        n_so = 2*n_spatial                        # number of spin-orbitals
        n_atoms = len(qc_mol.atom)                # should be 2 here
        print(f"                        Basis functions = {n_so//n_atoms} spin-orbitals per atom, {n_so} total")
        param_gates = {gates.RY, gates.RZ, gates.RX}
        # ── count single- and two-qubit gates ──
        ops = circ.queue      # the list of Gate objects in your circuit
        param_indices = [i for i, op in enumerate(ops) if type(op) in param_gates]
        n_single = sum(1 for gate in ops if len(gate.qubits) == 1)
        n_two    = sum(1 for gate in ops if len(gate.qubits) == 2)
        print(f"                        Number of qubits = {nqubits}")
        print(f"                        Number of circuit gates = "
              f"{n_single} single-qubit, {n_two} two-qubit")
        print(f"                        Number of circuit parameters         = {nparams}")
        print(circ.summary())
        sys.stdout.flush()
        #SleepForever()

    # Define initial amplitudes
    if init_params is None:
        # Cold start: choose HF (zero), MP2 (from ansatz), or random angles
        if INITAMPLITUDES == "HF":
            init_params = np.zeros(nparams)
        elif INITAMPLITUDES == "MP2" and ANSATZ == "UCCSD":
            init_params = np.asarray(circ.get_parameters()).flatten()
        else:
            init_params = np.random.uniform(0, 2 * np.pi, nparams)
    # Define the objective: expectation value of H at params
    def vqe_objective(params):
        # returns a Python float; backend uses state vectors (since no density matrix is explicitly passed and no noise is simulated)
        circ.set_parameters(params)
        return expectation(circ, H)




    if DEBUG:
        print(" --- 5) Classical optimization loop")
        sys.stdout.flush()
    if DEBUG and abs(raw_d - xStart)<1e-6:
        # Iteration counter for the callback
        iter_count = {'n': 0}
        def scipy_cb(xk):
            iter_count['n'] += 1
            # time one evaluation of the objective
            t0 = time.perf_counter()
            ek = vqe_objective(xk)
            t1 = time.perf_counter()
            dt = t1 - t0
            print(f"Iter {iter_count['n']:2d}: energy = {ek:.8f} Ha, params[:3] = {xk[:3]}..., eval_time(obj_func) = {dt:.3f} s")
            sys.stdout.flush()
        callback = scipy_cb
    else:
        callback = None
    # Now call SciPy’s minimize (instead of vqe.minimize):
    for r in range(REPEAT):
        if REPEAT>1:
            init_params = np.random.uniform(0, 2 * np.pi, nparams)
        test = minimize(
            vqe_objective,
            init_params,
            method=OPTIMIZER,
            tol=TOLERANCE,
            options={"maxiter": MAXITER, "disp": True},
            callback=callback
        )
        if r==0:
            result = test
        elif test.fun < result.fun:
            result = test
    # Extract best energy and parameters
    best_params = result.x
    best_energy = result.fun
    if DEBUG:
        print("Energy computed.")
        sys.stdout.flush()

    return scf_energy, best_energy, fci_energy, ccsd_t_energy, best_params


def main():
    # Record wall-clock time for performance metrics
    start = time.perf_counter()
    np.random.seed(0)  # reproducible random guesses

    # Generate bond-length grid by concatenating piecewise linspaces
    segments = []
    # for each (start,end,count), generate a linspace *excluding* the endpoint
    for (x0, x1), count in zip(zip(xPoints[:-1], xPoints[1:]), NPTS_LIST):
        seg = np.linspace(x0, x1, count, endpoint=False)
        segments.append(seg)
    # now replace the last segment so it *does* include xEnd as its endpoint
    last_start, last_end = xPoints[-2], xPoints[-1]
    last_count = NPTS_LIST[-1]
    segments[-1] = np.linspace(last_start, last_end, last_count, endpoint=True)
    distances = np.concatenate(segments)
    NPTS = distances.size

    TotalNPTS = sum(NPTS_LIST)
    progress = 0
    print(f"\n ^^^^^^^^ PROGRESS: {progress}/{TotalNPTS} ^^^^^^^^\n")

    # Decide between serial (warm start) or parallel (cold start)
    if THREADS == 1:
        scf_energies = []
        vqe_energies = []
        fci_energies = []
        ccsd_t_energies = []
        last_params = None
        for d in distances:
            scf_e, vqe_e, fci_e, ccsd_t_e, last_params = compute_energy(d, init_params=last_params)
            # convert units for reporting
            scf_e *= convert_energy
            vqe_e *= convert_energy
            fci_e *= convert_energy
            ccsd_t_e  *= convert_energy
            scf_energies.append(scf_e)
            vqe_energies.append(vqe_e)
            fci_energies.append(fci_e)
            ccsd_t_energies.append(ccsd_t_e)
            progress += 1
            print(f"\n ^^^^^^^^PROGRESS: {progress}/{TotalNPTS}\n ")
            logging.info(f"#  distance({d:8.4f})    scf_energy({scf_e:12.8f})    vqe_energy({vqe_e:12.8f})    fci_energy({fci_e:12.8f})")
        results = list(zip(scf_energies, vqe_energies, fci_energies, ccsd_t_energies))
    else:
        # Parallel runs: no warm-start dependence
        results = [None] * len(distances)
        with ProcessPoolExecutor(max_workers=THREADS) as executor:
            futures = {
                executor.submit(compute_energy, d): idx
                for idx, d in enumerate(distances)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                scf_e, vqe_e, fci_e, ccsd_t_e, _ = fut.result()
                # convert units
                scf_e *= convert_energy
                vqe_e *= convert_energy
                fci_e *= convert_energy
                ccsd_t_e  *= convert_energy
                results[idx] = (scf_e, vqe_e, fci_e, ccsd_t_e)
                progress += 1
                print(f"\n ^^^^^^^^PROGRESS: {progress}/{TotalNPTS}\n ")
        # now log in ascending-distance order
        for idx, (scf_e, vqe_e, fci_e, ccsd_t_e) in enumerate(results):
            logging.info(f"{distances[idx]:8.4f}    {scf_e:12.8f}    {vqe_e:12.8f}    {fci_e:12.8f}    {ccsd_t_e:12.8f}")


    # --- Plot results: Hartree–Fock vs VQE dissociation curves ---
    scf_energies = [res[0] for res in results]
    vqe_energies = [res[1] for res in results]
    fci_energies = [res[2] for res in results]
    ccsd_t_energies = [res[3] for res in results]
    plt.plot(distances, scf_energies, 'o', color='blue', label='HF')
    plt.plot(distances, vqe_energies, '--', color='orange', label=f"VQE [{BASIS}]")
    plt.plot(distances, fci_energies, '-', color='lightgreen', label=f"FCI [{FCIBASIS}]")
    plt.plot(distances, ccsd_t_energies, ':', color='black', label=f"CCSD(T) [{CCSDTBASIS}]")

    # Annotate minimum VQE energy point
    idx_min = np.argmin(vqe_energies)
    min_d = distances[idx_min]
    min_e = vqe_energies[idx_min]
    plt.scatter(min_d, min_e, marker='*', color='red', s=80, zorder=5)
    plt.annotate(
        f"VQE: {{ {min_e:.4f} {energy_unit} @ {min_d:.4f} {length_unit} }}",
        xy=(min_d, min_e),
        xytext=(min_d + 0.1*(distances[-1]-distances[0]),
                min_e + 0.05*(max(vqe_energies)-min_e)),
        color='black',
        arrowprops=dict(arrowstyle='->',color='black')
    )

    plt.legend()
    plt.xlabel(f"distance [{length_unit}]")
    plt.ylabel(f"energy [{energy_unit}]")
    plt.title(f"{PROJECT}")
    plt.grid(True)
    plt.tight_layout()

    # === Total runtime ===
    elapsed = time.perf_counter() - start
    print(f"\nTotal wall-clock time: {elapsed:.2f} seconds")
    # Save figure with timestamp and parameter summary
    ANSATZstring = f"{ANSATZ}({ANSATZPARAMS})"
    plt.savefig(os.path.join(data_dir, f"qcqc_{PROJECT}_{timestamp}_[{xStart}-{xEnd}]({NPTS})_{ANSATZstring}_{INITAMPLITUDES}_{BASIS}_FCI={FCIBASIS}_CCSDT={CCSDTBASIS}_{OPTIMIZER}_{elapsed:.0f}sec.pdf"))
    plt.show()


if __name__ == "__main__":
    # 1) Multiprocessing start method
    mp.set_start_method("spawn", force=True)

    # 2) Now configure your logger to write .dat and stream to the Tee’d stdout
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(os.path.join(data_dir, f"qcqc_{PROJECT}_{timestamp}.dat"), mode="w"),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # 3) Self-backup and its print() now goes through the Tee
    if mp.current_process().name == "MainProcess":
        script_path = os.path.realpath(__file__)
        backup_name = os.path.join(data_dir, f"qcqc_{PROJECT}_{timestamp}.py")
        shutil.copy(script_path, backup_name)
        print(f"Created backup of script as {backup_name}")

    # 4) Finally run the main computation
    main()
