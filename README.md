# Expresibilidad y capacidad de entrelazamiento (Experimento E6)

Notebooks que cuantifican la calidad intrínseca de los circuitos parametrizados (PQC)
del pipeline de clasificación de ruido mixto del TFM, con los descriptores de:

> Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019). *Expressibility and entangling
> capability of parameterized quantum circuits for hybrid quantum-classical algorithms*.
> **Adv. Quantum Technol.** 2, 1900070. (arXiv:1905.10876)

Marco de encoding/feature maps/kernels según el curso
[IBM Quantum Machine Learning](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/introduction).

## Métricas

- **Expresibilidad** = `D_KL( P_PQC(F) || P_Haar(F) )`, con `F=|<psi(θ)|psi(φ)>|²` y
  `P_Haar(F)=(N-1)(1-F)^(N-2)`, `N=2^n`. **KL menor ⇒ más expresivo.**
- **Capacidad de entrelazamiento (Meyer-Wallach)** `Q = (2/n)·Σ_k(1 - Tr ρ_k²)`,
  promediada sobre parámetros. `Q∈[0,1]`.

Ambas implementadas en `qexpr.py` con solo `qiskit.quantum_info` (sin dependencias extra).

## Contenido

| Fichero | Qué hace |
|---|---|
| `qexpr.py` | Módulo con `build_pqc`, `sample_fidelities`, `expressibility_kl`, `meyer_wallach_Q`, `descriptors`, `paper_circuit` y ayudas de figura. |
| `01_expressibility_entangling.ipynb` | Métricas del circuito de producción + grid E6 (`{zz,pauli}×{real_amplitudes,efficient_su2}×{4,6q}`) + saturación con `reps`. |
| `02_paper_reference_circuits.ipynb` | Reproduce un subconjunto de los 19 circuitos del paper y valida el ranking (Q≈0 para el circuito sin entanglement, CRX≻CRZ). |
| `03_metric_vs_accuracy.ipynb` | Entrena la HQNN (CNN+EstimatorQNN) por familia de circuito y correlaciona accuracy con expresibilidad y Q. |

## Entorno

Requiere **Qiskit 2.3 + qiskit-machine-learning 0.9** (notebooks 01–02) y además
**torch + torchvision** (notebook 03), tal como en `../requirements.txt` /
`../../TFM-EXPERIMENTS/requirements.txt`.

> Nota: el `../.venv` incluido apunta a un intérprete de otra máquina y no es
> reutilizable aquí; crea un entorno nuevo (p. ej. `py -3.12 -m venv .venv` y
> `pip install -r ../../TFM-EXPERIMENTS/requirements.txt`) o usa tu kernel habitual.

## Salidas

Se escriben con la convención estable `E6_*` en `../../TFM-EXPERIMENTS/outputs/`
(la memoria `.tex` las referencia). `OUTPUT_DIR` es configurable al inicio de cada notebook.

- Tablas: `E6_circuit_descriptors.csv`, `E6_expressibility.csv`, `E6_entangling.csv`,
  `E6_paper_reference.csv`, `E6_metric_vs_accuracy.csv`.
- Figuras (300 dpi): `E6_expressibility_bars.png`, `E6_entangling_bars.png`,
  `E6_expressibility_vs_reps.png`, `E6_fidelity_histograms.png`,
  `E6_paper_reference.png`, `E6_expr_vs_accuracy.png`, `E6_ent_vs_accuracy.png`.

## Parámetros de ejecución

`N_SAMPLES` (muestras de fidelidad; el paper usa ~5000, por defecto 2000–3000 para
rapidez) y, en el notebook 03, `EPOCHS` y `CONFIGS`. Súbelos para las cifras finales.

## Verificación realizada

`qexpr.py` y la lógica de 01/02 se probaron en un entorno Qiskit 2.3.0 + qml 0.9.0:
- Circuito de producción (zz+RealAmplitudes 4q): `D_KL≈0.06`, `Q≈0.83`.
- Circuito 1 (sin entanglement): `Q=0.000`; CRX (14) más expresivo y más entrelazante que CRZ (13).
- Las 6 QNN del notebook 03 se construyen y hacen `forward` con salida de 3 clases.
