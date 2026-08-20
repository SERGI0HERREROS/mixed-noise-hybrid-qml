# experiment-1 — Primer experimento: expresibilidad + testbed HQVAE

Primera rama experimental del TFM. Reúne dos bloques de trabajo desarrollados
en paralelo sobre el pipeline común de clasificación de ruido mixto en MNIST
(Gaussiano, Sal y Pimienta, Speckle) con modelos híbridos cuántico-clásicos:

1. **Expresibilidad y capacidad de entrelazamiento de circuitos (Experimento E6)**.
2. **Testbed de reconstrucción (denoising) híbrida con HQVAE**, incluyendo un
   primer prototipo combinado de clasificación + reconstrucción.

## 1. Expresibilidad y capacidad de entrelazamiento (E6)

Notebooks que cuantifican la calidad intrínseca de los circuitos parametrizados
(PQC) usados como capa cuántica, con los descriptores de:

> Sim, S., Johnson, P. D., & Aspuru-Guzik, A. (2019). *Expressibility and entangling
> capability of parameterized quantum circuits for hybrid quantum-classical algorithms*.
> **Adv. Quantum Technol.** 2, 1900070. (arXiv:1905.10876)

Marco de encoding/feature maps/kernels según el curso
[IBM Quantum Machine Learning](https://quantum.cloud.ibm.com/learning/en/courses/quantum-machine-learning/introduction).

### Métricas

- **Expresibilidad** = `D_KL( P_PQC(F) || P_Haar(F) )`, con `F=|<psi(θ)|psi(φ)>|²` y
  `P_Haar(F)=(N-1)(1-F)^(N-2)`, `N=2^n`. **KL menor ⇒ más expresivo.**
- **Capacidad de entrelazamiento (Meyer-Wallach)** `Q = (2/n)·Σ_k(1 - Tr ρ_k²)`,
  promediada sobre parámetros. `Q∈[0,1]`.

Ambas implementadas en `qexpr.py` con solo `qiskit.quantum_info` (sin dependencias extra).

### Contenido

| Fichero | Qué hace |
|---|---|
| `qexpr.py` | Módulo con `build_pqc`, `sample_fidelities`, `expressibility_kl`, `meyer_wallach_Q`, `descriptors`, `paper_circuit` y ayudas de figura. |
| `01_expressibility_entangling.ipynb` | Métricas del circuito de producción + grid E6 (`{zz,pauli}×{real_amplitudes,efficient_su2}×{4,6q}`) + saturación con `reps`. |
| `02_paper_reference_circuits.ipynb` | Reproduce un subconjunto de los 19 circuitos del paper y valida el ranking (Q≈0 para el circuito sin entanglement, CRX≻CRZ). |
| `03_metric_vs_accuracy.ipynb` | Entrena la HQNN (CNN+EstimatorQNN) por familia de circuito y correlaciona accuracy con expresibilidad y Q. |

### Verificación realizada

`qexpr.py` y la lógica de 01/02 se probaron en un entorno Qiskit 2.3.0 + qml 0.9.0:
- Circuito de producción (zz+RealAmplitudes 4q): `D_KL≈0.06`, `Q≈0.83`.
- Circuito 1 (sin entanglement): `Q=0.000`; CRX (14) más expresivo y más entrelazante que CRZ (13).
- Las 6 QNN del notebook 03 se construyen y hacen `forward` con salida de 3 clases.

## 2. Prototipo combinado: clasificación + reconstrucción (`Hybrid CQNN-VQAE.ipynb`)

Notebook exploratorio que replica el flujo clásico completo (clasificación de
ruido + *denoising*) sustituyendo parte del procesamiento por una QNN:

- **Clasificación de ruido mixto** con una HQNN (CNN + `EstimatorQNN` de 4
  qubits, `zz_feature_map` + `real_amplitudes`). Esta sección se dejó como
  borrador (celdas comentadas) a la espera de fijar observables y `reps`; se
  incluye a cambio una red **clásica equivalente** ya entrenada como referencia
  (accuracy de test = 1.0 en 10 épocas).
- **Denoising con `ConvHQVAE`**: encoder/decoder convolucional + bloque
  cuántico (`EstimatorQNN` de 4 qubits, `pauli_feature_map` + `real_amplitudes`,
  6 observables) en el espacio latente, con truco de reparametrización y
  pérdida β-VAE (`BCE + β·KL`, β=0.5). Entrenado 15 épocas sobre 300/100/100
  muestras; evaluado con **MSE** (≈0.066) y **PSNR** (≈11.8 dB). El cálculo de
  **SSIM** quedó pendiente por falta de la dependencia `scikit-image` en el
  entorno de ejecución.

Este notebook es el punto de partida que después se separa en dos testbeds
dedicados (`test-hcqnn/` para clasificación, `test-hqvae/` para reconstrucción)
en ramas posteriores.

## 3. Testbed de reconstrucción — `test-hqvae/`

Sandbox para lanzar pruebas sistemáticas de denoising con HQVAE frente a un
ConvVAE clásico, variando el circuito cuántico (qubits, feature map, ansatz,
entrelazamiento, reps) y el optimizador. Cada ejecución se guarda como HTML
autocontenido en `test-hqvae/results/`.

| Fichero | Qué es |
|---|---|
| `test-hqvae/hqvae_lib.py` | Código reutilizable: ruido mixto, dataset de denoising, `ConvVAE`, `ConvHQVAE`, entrenamiento, métricas, informes HTML y el barrido (`default_sweep`). |
| `test-hqvae/test_hqvae.ipynb` | Notebook driver: setup → datos → run único → barrido (4 y 6 qubits) → comparación de optimizadores → export nbconvert. |
| `test-hqvae/README.md` | Documentación detallada del testbed (arquitectura, barrido por defecto, convención de nombres). |
| `test-hqvae/results/` | Resultados del barrido inicial: 6 configuraciones de circuito (`HQVAE4_PFM_RA_linear_r1`, `HQVAE4_ZZFM_RA_linear_r1`, `HQVAE6_ZZFM_ESU2_linear_r1`, `HQVAE6_ZZFM_RA_full_r1`, `HQVAE6_ZZFM_RA_linear_r1`, `HQVAE6_ZZFM_RA_linear_r2`) + comparación de optimizadores (`cobyla`, `rmsprop`, `sgd`, `spsa`) sobre la configuración base, más `sweep_metrics.csv` con las métricas agregadas. |

## Historial de esta rama

- **`first experiment finished`**: commit único que introduce todo lo anterior
  — módulo y notebooks de expresibilidad (E6), el prototipo combinado
  `Hybrid CQNN-VQAE.ipynb`, y el testbed `test-hqvae/` con su primer barrido
  de circuitos y optimizadores.

## Entorno

Requiere **Qiskit 2.3 + qiskit-machine-learning 0.9** (notebooks 01–02) y además
**torch + torchvision** (notebook 03 y los testbeds), tal como en
`requirements.txt`. Para las métricas SSIM del testbed HQVAE se necesita
además `scikit-image` (ya listado en `requirements.txt`, pero ausente en el
entorno donde se ejecutó `Hybrid CQNN-VQAE.ipynb`).

> Nota: el `.venv` incluido apunta a un intérprete de otra máquina y no es
> reutilizable aquí; crea un entorno nuevo (p. ej. `py -3.12 -m venv .venv` y
> `pip install -r requirements.txt`) o usa tu kernel habitual.

## Salidas

Los notebooks 01–03 escriben con la convención estable `E6_*` en
`../TFM-EXPERIMENTS/outputs/` (la memoria `.tex` las referencia). `OUTPUT_DIR`
es configurable al inicio de cada notebook.

- Tablas: `E6_circuit_descriptors.csv`, `E6_expressibility.csv`, `E6_entangling.csv`,
  `E6_paper_reference.csv`, `E6_metric_vs_accuracy.csv`.
- Figuras (300 dpi): `E6_expressibility_bars.png`, `E6_entangling_bars.png`,
  `E6_expressibility_vs_reps.png`, `E6_fidelity_histograms.png`,
  `E6_paper_reference.png`, `E6_expr_vs_accuracy.png`, `E6_ent_vs_accuracy.png`.

## Parámetros de ejecución

`N_SAMPLES` (muestras de fidelidad; el paper usa ~5000, por defecto 2000–3000 para
rapidez) y, en el notebook 03, `EPOCHS` y `CONFIGS`. Súbelos para las cifras finales.
