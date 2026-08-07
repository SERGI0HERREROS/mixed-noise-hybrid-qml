"""qexpr.py — Métricas de expresibilidad y capacidad de entrelazamiento de
circuitos cuánticos parametrizados (PQC) para el TFM de clasificación de ruido mixto.

Implementa los dos descriptores de:
    Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019).
    "Expressibility and entangling capability of parameterized quantum circuits
     for hybrid quantum-classical algorithms". Adv. Quantum Technol. 2, 1900070.
    (arXiv:1905.10876)

Descriptores:
  * Expresibilidad = D_KL( P_PQC(F) || P_Haar(F) ), con F = |<psi(theta)|psi(phi)>|^2
    y P_Haar(F) = (N-1)(1-F)^(N-2), N = 2^n.  KL menor  =>  más expresivo.
  * Capacidad de entrelazamiento (Meyer-Wallach):
        Q = (2/n) * sum_k (1 - Tr(rho_k^2)),  promediado sobre parámetros aleatorios.
    Q in [0, 1]; 0 = estado producto, 1 = máximamente entrelazado.

Compatible con Qiskit 2.x + qiskit-machine-learning 0.9 (mismas factorías de circuito
que usa el pipeline del TFM: zz_feature_map, pauli_feature_map, real_amplitudes,
efficient_su2). Solo usa qiskit.quantum_info (Statevector, partial_trace) — sin
dependencias extra.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import (
    zz_feature_map,
    pauli_feature_map,
    real_amplitudes,
    efficient_su2,
)
from qiskit.quantum_info import Statevector, partial_trace, state_fidelity


# --------------------------------------------------------------------------- #
# Construcción de circuitos del pipeline del TFM
# --------------------------------------------------------------------------- #
def build_pqc(
    feature_map: str | None = "zz",
    ansatz: str = "real_amplitudes",
    n_qubits: int = 4,
    reps: int = 1,
    fm_reps: int = 1,
    fm_entanglement: str = "full",
    ansatz_entanglement: str = "reverse_linear",
    paulis: list[str] | None = None,
    include_feature_map: bool = True,
):
    """Compone feature map + ansatz replicando los circuitos del TFM.

    Parameters
    ----------
    feature_map : {"zz", "pauli", None}
        Tipo de feature map. `None` o include_feature_map=False -> solo ansatz
        (útil para reproducir los circuitos del paper, que no tienen encoding).
    ansatz : {"real_amplitudes", "efficient_su2"}
    n_qubits, reps : dimensiones del ansatz.
    fm_reps, fm_entanglement : repeticiones/entanglement del feature map.
    ansatz_entanglement : entanglement del ansatz.
    paulis : lista de Paulis para pauli_feature_map (por defecto ['X','Y','ZZ']).

    Returns
    -------
    (QuantumCircuit, dict)  ->  circuito y metadatos (nº de params, profundidad...).
    """
    qc = QuantumCircuit(n_qubits)

    if include_feature_map and feature_map is not None:
        if feature_map == "zz":
            fm = zz_feature_map(n_qubits, reps=fm_reps, entanglement=fm_entanglement)
        elif feature_map == "pauli":
            fm = pauli_feature_map(
                n_qubits,
                reps=fm_reps,
                paulis=paulis or ["X", "Y", "ZZ"],
                entanglement=fm_entanglement,
            )
        else:
            raise ValueError(f"feature_map desconocido: {feature_map}")
        qc.compose(fm, inplace=True)

    if ansatz == "real_amplitudes":
        an = real_amplitudes(n_qubits, reps=reps, entanglement=ansatz_entanglement)
    elif ansatz == "efficient_su2":
        an = efficient_su2(n_qubits, reps=reps, entanglement=ansatz_entanglement)
    else:
        raise ValueError(f"ansatz desconocido: {ansatz}")
    qc.compose(an, inplace=True)

    info = {
        "feature_map": feature_map if include_feature_map else None,
        "ansatz": ansatz,
        "n_qubits": n_qubits,
        "reps": reps,
        "n_params": qc.num_parameters,
        "depth": qc.decompose().depth(),
    }
    return qc, info


# --------------------------------------------------------------------------- #
# Muestreo de fidelidades y métricas
# --------------------------------------------------------------------------- #
def sample_fidelities(qc: QuantumCircuit, n_samples: int = 5000, seed: int = 42) -> np.ndarray:
    """Muestrea `n_samples` fidelidades F=|<psi(theta)|psi(phi)>|^2 con theta,phi
    uniformes en [0, 2*pi). Devuelve un array de longitud n_samples."""
    n_params = qc.num_parameters
    if n_params == 0:
        raise ValueError("El circuito no tiene parámetros que muestrear.")
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, n_params))
    b = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, n_params))

    fids = np.empty(n_samples, dtype=float)
    for i in range(n_samples):
        sv1 = Statevector(qc.assign_parameters(a[i]))
        sv2 = Statevector(qc.assign_parameters(b[i]))
        fids[i] = float(np.real(state_fidelity(sv1, sv2)))
    return fids


def haar_bin_probs(n_qubits: int, bin_edges: np.ndarray) -> np.ndarray:
    """Probabilidad de Haar por bin, EXACTA (sin muestrear), vía la CDF
    P(F<=f) = 1 - (1-f)^(N-1) de P_Haar(F)=(N-1)(1-F)^(N-2), con N=2^n."""
    N = 2 ** n_qubits
    cdf = 1.0 - (1.0 - bin_edges) ** (N - 1)
    return np.diff(cdf)


def expressibility_kl(
    fidelities: np.ndarray, n_qubits: int, n_bins: int = 75, eps: float = 1e-12
):
    """Expresibilidad = D_KL(P_PQC || P_Haar) (KL menor => más expresivo).

    Returns
    -------
    (kl, p_pqc, p_haar, bin_edges)  para poder dibujar el histograma comparado.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts, _ = np.histogram(fidelities, bins=bin_edges)
    p_pqc = counts / counts.sum()
    p_haar = haar_bin_probs(n_qubits, bin_edges)

    mask = p_pqc > 0
    kl = float(np.sum(p_pqc[mask] * np.log(p_pqc[mask] / (p_haar[mask] + eps))))
    return kl, p_pqc, p_haar, bin_edges


def meyer_wallach_Q(qc: QuantumCircuit, n_samples: int = 5000, seed: int = 42):
    """Capacidad de entrelazamiento de Meyer-Wallach, promediada sobre parámetros.

    Returns
    -------
    (mean_Q, std_Q)  con Q in [0, 1].
    """
    n = qc.num_qubits
    n_params = qc.num_parameters
    if n_params == 0:
        raise ValueError("El circuito no tiene parámetros que muestrear.")
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.0, 2.0 * np.pi, size=(n_samples, n_params))

    q_vals = np.empty(n_samples, dtype=float)
    all_qubits = list(range(n))
    for i in range(n_samples):
        sv = Statevector(qc.assign_parameters(a[i]))
        purity_sum = 0.0
        for k in range(n):
            trace_out = all_qubits[:k] + all_qubits[k + 1:]
            rho_k = partial_trace(sv, trace_out)
            purity_sum += float(np.real(rho_k.purity()))
        # Q = (2/n) * sum_k (1 - Tr rho_k^2) = 2 * (1 - mean_k Tr rho_k^2)
        q_vals[i] = 2.0 * (1.0 - purity_sum / n)
    return float(np.mean(q_vals)), float(np.std(q_vals))


def descriptors(
    qc: QuantumCircuit,
    n_qubits: int,
    n_samples: int = 5000,
    n_bins: int = 75,
    seed: int = 42,
) -> dict:
    """Calcula de una vez expresibilidad (KL) y entrelazamiento (Q) de un circuito."""
    fids = sample_fidelities(qc, n_samples=n_samples, seed=seed)
    kl, _, _, _ = expressibility_kl(fids, n_qubits, n_bins=n_bins)
    q_mean, q_std = meyer_wallach_Q(qc, n_samples=n_samples, seed=seed)
    return {
        "n_qubits": n_qubits,
        "n_params": qc.num_parameters,
        "depth": qc.decompose().depth(),
        "expressibility_kl": kl,
        "entangling_Q": q_mean,
        "entangling_Q_std": q_std,
    }


# --------------------------------------------------------------------------- #
# Subconjunto representativo de los 19 circuitos del paper (Sim et al. 2019, Fig. 2)
# --------------------------------------------------------------------------- #
# Se reproducen sobre `n_qubits` (el paper los dibuja en 4). Cada circuito repite
# su bloque `reps` veces. Solo tienen ansatz (sin encoding de datos).
#
# Cobertura elegida (de baja a alta expresibilidad/entrelazamiento):
#   1  -> rotaciones locales, SIN entrelazamiento (referencia: Q ~ 0).
#   2  -> rotaciones + cadena de CNOT.
#   9  -> H + cadena de CZ + RX  (bloque tipo IQP).
#   13 -> RY + anillo de CRZ + RY + anillo de CRZ.
#   14 -> idéntico a 13 con CRX (el paper: CRX supera a CRZ).
#   15 -> RY + anillo de CNOT + RY + anillo de CNOT.
#   19 -> RX,RZ + anillo de CRX (alta expresibilidad y entrelazamiento).
PAPER_SUBSET = [1, 2, 9, 13, 14, 15, 19]


def paper_circuit(idx: int, n_qubits: int = 4, reps: int = 1) -> QuantumCircuit:
    """Constructor de un circuito del paper por su índice (ver PAPER_SUBSET)."""
    qc = QuantumCircuit(n_qubits)
    counter = [0]

    def th() -> Parameter:
        p = Parameter(f"θ{counter[0]}")
        counter[0] += 1
        return p

    for _ in range(reps):
        if idx == 1:
            # Rotaciones locales RX, RZ. Sin entrelazamiento.
            for q in range(n_qubits):
                qc.rx(th(), q)
                qc.rz(th(), q)

        elif idx == 2:
            # RX, RZ + cadena descendente de CNOT.
            for q in range(n_qubits):
                qc.rx(th(), q)
                qc.rz(th(), q)
            for q in range(n_qubits - 1, 0, -1):
                qc.cx(q, q - 1)

        elif idx == 9:
            # Hadamard en todos + cadena de CZ + RX (estructura tipo IQP).
            for q in range(n_qubits):
                qc.h(q)
            for q in range(n_qubits - 1):
                qc.cz(q, q + 1)
            for q in range(n_qubits):
                qc.rx(th(), q)

        elif idx == 13:
            # RY + anillo de CRZ + RY + anillo de CRZ (offset).
            for q in range(n_qubits):
                qc.ry(th(), q)
            for q in range(n_qubits):
                qc.crz(th(), q, (q - 1) % n_qubits)
            for q in range(n_qubits):
                qc.ry(th(), q)
            for q in range(n_qubits):
                qc.crz(th(), q, (q + 1) % n_qubits)

        elif idx == 14:
            # Igual que 13 pero con CRX (el paper: CRX > CRZ).
            for q in range(n_qubits):
                qc.ry(th(), q)
            for q in range(n_qubits):
                qc.crx(th(), q, (q - 1) % n_qubits)
            for q in range(n_qubits):
                qc.ry(th(), q)
            for q in range(n_qubits):
                qc.crx(th(), q, (q + 1) % n_qubits)

        elif idx == 15:
            # RY + anillo de CNOT + RY + anillo de CNOT (entangladores sin parámetro).
            for q in range(n_qubits):
                qc.ry(th(), q)
            for q in range(n_qubits):
                qc.cx(q, (q + 1) % n_qubits)
            for q in range(n_qubits):
                qc.ry(th(), q)
            for q in range(n_qubits):
                qc.cx(q, (q - 1) % n_qubits)

        elif idx == 19:
            # RX, RZ + anillo de CRX.
            for q in range(n_qubits):
                qc.rx(th(), q)
                qc.rz(th(), q)
            for q in range(n_qubits):
                qc.crx(th(), q, (q + 1) % n_qubits)

        else:
            raise ValueError(
                f"Circuito {idx} no implementado. Disponibles: {PAPER_SUBSET}"
            )

    return qc


# --------------------------------------------------------------------------- #
# Ayudas de figura
# --------------------------------------------------------------------------- #
def plot_fidelity_histogram(fidelities, n_qubits, ax=None, label=None, n_bins=75):
    """Dibuja el histograma de fidelidades del PQC frente a la distribución de Haar."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3.2))
    kl, p_pqc, p_haar, edges = expressibility_kl(fidelities, n_qubits, n_bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    ax.bar(centers, p_pqc, width=width, alpha=0.6,
           label=(label or "PQC") + f" (KL={kl:.3f})")
    ax.plot(centers, p_haar, "k--", lw=1.5, label="Haar")
    ax.set_xlabel("Fidelidad $F$")
    ax.set_ylabel("Probabilidad")
    ax.legend(fontsize=8)
    return ax, kl


def savefig(fig, path, dpi=300):
    """Guarda una figura con la convención del TFM (300 dpi, fondo blanco)."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
