"""
Quantum Approximate Optimization Algorithm (QAOA) using Qiskit.

Pipeline
--------
1. Build the QUBO matrix from the problem formulation.
2. Encode it as a Pauli-Z Ising Hamiltonian.
3. Construct a QAOA circuit of depth p (default p=2).
4. Optimise the variational parameters (β, γ) with COBYLA.
5. Sample the circuit on a statevector/QASM simulator.
6. Decode the best bit-string back to continuous decision variables.

The problem uses n_vars=6 decision variables × n_bits=3 bits each
→ 18-qubit QAOA circuit.
"""

from __future__ import annotations

import time
import warnings
import numpy as np
from typing import Callable

# Qiskit imports — guarded so the module can be imported even if Qiskit
# is not installed (a warning is emitted; QAOA simply won't run).
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter, ParameterVector
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from scipy.optimize import minimize as scipy_minimize
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    warnings.warn(
        "Qiskit or qiskit-aer is not installed. "
        "QAOA will fall back to classical simulation.",
        ImportWarning,
        stacklevel=2,
    )

from src.optimization.problem import AgricultureOptimizationProblem, ProblemConfig


class QAOAOptimizer:
    """
    QAOA-based optimiser for the agriculture resource allocation problem.

    Parameters
    ----------
    config   : Problem configuration (n_bits controls qubit count)
    p_layers : QAOA depth (number of (U_C, U_B) layers)
    n_shots  : Measurement shots for the QASM simulator
               Set to 0 to use statevector (exact, noiseless)
    max_iter : Classical optimizer max iterations (COBYLA)
    """

    def __init__(
        self,
        config: ProblemConfig | None = None,
        p_layers: int = 2,
        n_shots: int = 4096,
        max_iter: int = 200,
    ):
        self.problem   = AgricultureOptimizationProblem(config)
        self.p_layers  = p_layers
        self.n_shots   = n_shots
        self.max_iter  = max_iter

        self._best_solution:   np.ndarray | None = None
        self._best_fitness:    float = float("inf")
        self._convergence:     list[float] = []
        self._elapsed_seconds: float = 0.0
        self._n_qubits:        int   = 0

    # ------------------------------------------------------------------
    # Hamiltonian construction from QUBO
    # ------------------------------------------------------------------

    def _qubo_to_ising(
        self, Q: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Convert QUBO matrix Q to Ising {-1,+1} coupling J and bias h.

        x_i ∈ {0,1},  z_i = 2x_i - 1  →  x_i = (z_i + 1) / 2

        Q-QUBO objective: E = x^T Q x
        Ising objective : E_ising = sum_ij J_ij z_i z_j + sum_i h_i z_i + const
        """
        n    = Q.shape[0]
        J    = np.zeros((n, n))
        h    = np.zeros(n)
        offset = 0.0

        for i in range(n):
            for j in range(n):
                if i == j:
                    h[i]      += Q[i, i] / 2.0
                    offset    += Q[i, i] / 4.0
                elif i < j:
                    J[i, j]   += Q[i, j] / 4.0
                    h[i]      += Q[i, j] / 4.0
                    h[j]      += Q[i, j] / 4.0
                    offset    += Q[i, j] / 4.0

        return J, h, offset

    def _build_cost_operator(
        self, J: np.ndarray, h: np.ndarray, n: int
    ) -> list[tuple[str, float]]:
        """Build Pauli-Z cost operator terms [(pauli_str, coeff), ...]."""
        terms: list[tuple[str, float]] = []

        # Single-qubit Z terms
        for i in range(n):
            if abs(h[i]) > 1e-12:
                pauli = "I" * (n - 1 - i) + "Z" + "I" * i
                terms.append((pauli, h[i]))

        # Two-qubit ZZ terms
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    pauli = ["I"] * n
                    pauli[n - 1 - i] = "Z"
                    pauli[n - 1 - j] = "Z"
                    terms.append(("".join(pauli), J[i, j]))

        return terms

    # ------------------------------------------------------------------
    # QAOA circuit
    # ------------------------------------------------------------------

    def _build_qaoa_circuit(
        self,
        cost_terms: list[tuple[str, float]],
        n: int,
        gammas: list[Parameter],
        betas: list[Parameter],
    ) -> QuantumCircuit:
        """Construct QAOA circuit with p layers."""
        qc = QuantumCircuit(n)

        # Initial state: uniform superposition
        qc.h(range(n))

        for layer in range(self.p_layers):
            gamma = gammas[layer]
            beta  = betas[layer]

            # --- Cost unitary U_C(γ) ---
            for pauli_str, coeff in cost_terms:
                active_qubits = [
                    n - 1 - i
                    for i, p in enumerate(pauli_str)
                    if p == "Z"
                ]
                if len(active_qubits) == 1:
                    qc.rz(2 * gamma * coeff, active_qubits[0])
                elif len(active_qubits) == 2:
                    q0, q1 = active_qubits
                    qc.cx(q0, q1)
                    qc.rz(2 * gamma * coeff, q1)
                    qc.cx(q0, q1)

            # --- Mixer unitary U_B(β) ---
            for q in range(n):
                qc.rx(2 * beta, q)

        qc.measure_all()
        return qc

    # ------------------------------------------------------------------
    # Expectation value evaluation
    # ------------------------------------------------------------------

    def _expectation(
        self,
        params: np.ndarray,
        cost_terms: list[tuple[str, float]],
        n: int,
        gammas_sym: list[Parameter],
        betas_sym: list[Parameter],
        simulator: "AerSimulator",
    ) -> float:
        """Evaluate ⟨C⟩ for given (γ, β) parameters."""
        qc = self._build_qaoa_circuit(cost_terms, n, gammas_sym, betas_sym)

        # Bind parameters
        param_dict = {}
        for k in range(self.p_layers):
            param_dict[gammas_sym[k]] = float(params[k])
            param_dict[betas_sym[k]]  = float(params[self.p_layers + k])

        bound_qc = qc.assign_parameters(param_dict)

        # Run on simulator
        job    = simulator.run(bound_qc, shots=self.n_shots)
        counts = job.result().get_counts()

        # Compute expectation value from sampled bit-strings
        total_shots = sum(counts.values())
        expectation = 0.0
        for bitstring, count in counts.items():
            bits = np.array([int(b) for b in reversed(bitstring)], dtype=int)
            # Evaluate QUBO directly on bit-string
            cost = 0.0
            for pauli_str, coeff in cost_terms:
                zvals = [
                    1 - 2 * bits[n - 1 - i]
                    for i, p in enumerate(pauli_str)
                    if p == "Z"
                ]
                if zvals:
                    cost += coeff * float(np.prod(zvals))
            expectation += (count / total_shots) * cost

        return expectation

    # ------------------------------------------------------------------
    # Classical fallback (when Qiskit unavailable)
    # ------------------------------------------------------------------

    def _classical_fallback(self) -> dict:
        """
        Fallback: use differential evolution to solve the problem classically
        and wrap it with QAOA-style reporting.
        """
        warnings.warn(
            "Qiskit not available — running classical fallback for QAOA slot.",
            RuntimeWarning,
        )
        from scipy.optimize import differential_evolution
        cfg    = self.problem.config
        bounds = list(zip(cfg.lower_bounds, cfg.upper_bounds))
        history: list[float] = []

        def cb(xk, convergence):
            history.append(self.problem.objective(xk))

        res = differential_evolution(
            self.problem.objective,
            bounds=bounds,
            seed=42,
            maxiter=self.max_iter,
            tol=1e-10,
            callback=cb,
        )
        self._best_solution  = res.x
        self._best_fitness   = res.fun
        self._convergence    = history
        return res.x, res.fun, history

    # ------------------------------------------------------------------
    # Main optimisation
    # ------------------------------------------------------------------

    def optimise(self, verbose: bool = True) -> dict:
        """Run QAOA and return best solution summary."""
        t0 = time.perf_counter()

        if not QISKIT_AVAILABLE:
            best_x, best_val, history = self._classical_fallback()
        else:
            best_x, best_val, history = self._run_qaoa(verbose)

        self._elapsed_seconds = time.perf_counter() - t0
        self._best_solution   = best_x
        self._best_fitness    = best_val
        self._convergence     = history

        result = self.problem.summarise(self._best_solution, algorithm="QAOA")
        result["convergence"]     = self._convergence
        result["elapsed_seconds"] = round(self._elapsed_seconds, 3)
        result["p_layers"]        = self.p_layers
        result["n_qubits"]        = self._n_qubits
        result["qiskit_used"]     = QISKIT_AVAILABLE

        if verbose:
            print(f"\n  QAOA finished in {self._elapsed_seconds:.2f}s")
            print(f"  Best objective: {self._best_fitness:.6f}")
            print(f"  Predicted yield: {result['predicted_yield_t_ha']} t/ha")
            print(f"  Water usage: {result['irrigation_water_l_ha']} L/ha")
            if QISKIT_AVAILABLE:
                print(f"  Qubits used: {self._n_qubits}  |  QAOA depth p={self.p_layers}")

        return result

    def _run_qaoa(self, verbose: bool) -> tuple[np.ndarray, float, list[float]]:
        """Core QAOA execution on Qiskit Aer simulator."""
        cfg = self.problem.config

        if verbose:
            print(
                f"\n  Building QUBO matrix "
                f"({cfg.n_vars} vars × {cfg.n_bits} bits = "
                f"{cfg.n_vars * cfg.n_bits} qubits)..."
            )

        Q = self.problem.build_qubo_matrix()
        n = Q.shape[0]
        self._n_qubits = n

        J, h, offset = self._qubo_to_ising(Q)
        cost_terms   = self._build_cost_operator(J, h, n)

        # Symbolic parameters
        gammas_sym = [Parameter(f"gamma_{k}") for k in range(self.p_layers)]
        betas_sym  = [Parameter(f"beta_{k}")  for k in range(self.p_layers)]

        simulator = AerSimulator(method="statevector")

        history: list[float] = []

        def objective(params: np.ndarray) -> float:
            val = self._expectation(
                params, cost_terms, n, gammas_sym, betas_sym, simulator
            )
            history.append(val)
            return val

        # Initial parameters: random in [0, π] for γ, [0, π/2] for β
        rng    = np.random.default_rng(42)
        x0     = np.concatenate([
            rng.uniform(0, np.pi,     self.p_layers),  # gammas
            rng.uniform(0, np.pi / 2, self.p_layers),  # betas
        ])

        if verbose:
            print(f"  Running COBYLA with max_iter={self.max_iter}...")

        res = scipy_minimize(
            objective,
            x0,
            method="COBYLA",
            options={"maxiter": self.max_iter, "rhobeg": 0.5},
        )

        optimal_params = res.x

        # Final sample with more shots to find best bit-string
        qc = self._build_qaoa_circuit(cost_terms, n, gammas_sym, betas_sym)
        param_dict = {}
        for k in range(self.p_layers):
            param_dict[gammas_sym[k]] = float(optimal_params[k])
            param_dict[betas_sym[k]]  = float(optimal_params[self.p_layers + k])
        bound_qc = qc.assign_parameters(param_dict)

        final_shots = max(self.n_shots * 2, 8192)
        job    = simulator.run(bound_qc, shots=final_shots)
        counts = job.result().get_counts()

        # Find bit-string with lowest QUBO energy
        best_bits = None
        best_val  = float("inf")
        for bitstring in counts:
            # Strip any whitespace/separator characters Qiskit may add
            clean = bitstring.replace(" ", "")
            try:
                bits = np.array([int(b) for b in reversed(clean)], dtype=int)
            except ValueError:
                continue
            # Pad or trim to expected length if needed
            expected_len = cfg.n_vars * cfg.n_bits
            if len(bits) < expected_len:
                bits = np.pad(bits, (0, expected_len - len(bits)))
            elif len(bits) > expected_len:
                bits = bits[:expected_len]
            x   = self.problem.decode_from_binary(bits)
            val = self.problem.objective(x)
            if val < best_val:
                best_val  = val
                best_bits = bits

        # Safety fallback: if no valid bit-string was found use midpoint solution
        if best_bits is None:
            warnings.warn(
                "QAOA: no valid bit-string found in measurement counts. "
                "Falling back to midpoint solution.",
                RuntimeWarning,
            )
            best_x   = (cfg.lower_bounds + cfg.upper_bounds) / 2.0
            best_val = self.problem.objective(best_x)
        else:
            best_x = self.problem.decode_from_binary(best_bits)

        return best_x, best_val, history
