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

| Fichero | Qué hace | Estado |
|---|---|---|
| `qexpr.py` | Módulo con `build_pqc`, `sample_fidelities`, `expressibility_kl`, `meyer_wallach_Q`, `descriptors`, `paper_circuit` y ayudas de figura. | — |
| `01_expressibility_entangling.ipynb` | Métricas del circuito de producción + grid E6 (`{zz,pauli}×{real_amplitudes,efficient_su2}×{4,6q}`) + saturación con `reps`. | **ejecutado por completo** |
| `02_paper_reference_circuits.ipynb` | Reproduce un subconjunto de los 19 circuitos del paper y valida el ranking. | **ejecutado por completo** |
| `03_metric_vs_accuracy.ipynb` | Entrena la HQNN (CNN+EstimatorQNN) por familia de circuito y correlaciona accuracy con expresibilidad y Q. | **ejecutado**, pero no re-ejecutable (ver aviso) |

### Resultados ejecutados

Las cifras siguientes se leen directamente de las salidas guardadas en los
notebooks. Entorno de ejecución: **Google Colab**, Python 3.12, **Qiskit 2.5.0
+ qiskit-machine-learning 0.9.0**.

#### Circuito de producción (`zz` + RealAmplitudes, 4 qubits, `reps=1`)

`n_params = 12`, `depth = 22`, **`D_KL = 0.008928`**, **`Q = 0.834993 ± 0.101534`**
(`N_SAMPLES = 2000`, `SEED` fijada).

#### Rejilla E6 completa (`01_expressibility_entangling.ipynb`)

| Circuito | n_params | depth | `D_KL` | `Q` |
|---|---|---|---|---|
| zz + RealAmplitudes (4q) | 12 | 22 | 0.008928 | 0.834993 ± 0.101534 |
| zz + RealAmplitudes (6q) | 18 | 36 | 0.001004 | 0.949018 ± 0.039754 |
| zz + EfficientSU2 (4q) | 20 | 24 | 0.007948 | 0.830142 ± 0.104298 |
| zz + EfficientSU2 (6q) | 30 | 38 | 0.002525 | 0.949953 ± 0.036427 |
| pauli + RealAmplitudes (4q) | 12 | 31 | 0.020220 | 0.630375 ± 0.177257 |
| pauli + RealAmplitudes (6q) | 18 | 45 | 0.024332 | 0.761830 ± 0.125084 |
| pauli + EfficientSU2 (4q) | 20 | 33 | 0.007032 | 0.698853 ± 0.163617 |
| pauli + EfficientSU2 (6q) | 30 | 47 | 0.013429 | 0.800506 ± 0.111760 |

Los mapas `zz` dominan a los `pauli` en las dos métricas a igualdad de qubits, y
pasar de 4 a 6 qubits mejora ambas de forma sistemática.

**Saturación con la profundidad** (`reps` de 1 a 5), midiendo el ansatz aislado y
el circuito completo:

| `reps` | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `D_KL` (solo ansatz) | 0.310 | 0.181 | 0.131 | 0.110 | **0.141** |
| `D_KL` (circuito completo) | 0.009 | 0.007 | 0.009 | 0.008 | 0.011 |

La expresibilidad del ansatz mejora al añadir repeticiones pero deja de hacerlo
a partir de `reps=4`: es el efecto de saturación que describen Sim et al., y el
argumento por el que la configuración de producción usa `reps=1` (el circuito
completo ya está saturado desde la primera repetición, porque el feature map
aporta la mayor parte de la expresibilidad).

#### Validación contra el paper (`02_paper_reference_circuits.ipynb`)

Reimplementa 7 de los 19 circuitos de Sim et al. (`N_SAMPLES = 3000`):

| Circuito | n_params | depth | `D_KL` | `Q` |
|---|---|---|---|---|
| 14 (CRX) | 16 | 37 | 0.026054 | 0.547 |
| 13 (CRZ) | 16 | 28 | 0.058671 | 0.407 |
| 19 | 12 | 23 | 0.082167 | 0.417 |
| 15 | 8 | 10 | 0.186634 | 0.695 |
| 1 (sin entanglement) | 8 | 2 | 0.311468 | ≈0 (−4.5e−18) |
| 2 | 8 | 5 | 0.311468 | 0.625 |
| 9 | 4 | 9 | 0.684732 | 1.000 |

Resultado del bloque de comprobaciones: **4 / 4 superadas** — el circuito 1 no
entrelaza (`Q ≈ 0`); CRX (14) entrelaza más que CRZ (13); CRX es más expresivo
que CRZ; y CRX queda entre los tres más expresivos. El circuito 9 ilustra el
caso extremo de `Q = 1` con expresibilidad baja: entrelaza al máximo pero sus
4 parámetros no cubren el espacio de estados.

Este notebook es lo que legitima usar `qexpr.py` como instrumento de medida: sin
él, los descriptores de la rejilla anterior serían números sin contraste externo.

#### Descriptores frente a accuracy (`03_metric_vs_accuracy.ipynb`)

Entrena un clasificador híbrido por familia de circuito —8 épocas, Adam a `1e-3`,
semilla 42, 4 qubits, mismos observables y mismo extractor convolucional— y calcula
los descriptores sobre el mismo objeto `QuantumCircuit` que usa la red:

| Circuito | accuracy | `D_KL` | `Q` | n_params | depth |
|---|---|---|---|---|---|
| ZZFM + RA (rev_lin, r1) *[producción]* | 0.93 | 0.008928 | 0.834993 | 12 | 22 |
| ZZFM + RA (full, r1) | 0.33 | 0.008928 | 0.834993 | 12 | 23 |
| ZZFM + RA (rev_lin, r2) | 0.97 | 0.006976 | 0.832146 | 16 | 25 |
| ZZFM + ESU2 (rev_lin, r1) | 0.68 | 0.007948 | 0.830142 | 20 | 24 |
| PFM + RA (rev_lin, r1) | 0.34 | 0.020220 | 0.630375 | 12 | 31 |
| PFM + ESU2 (rev_lin, r1) | 1.00 | 0.007032 | 0.698853 | 20 | 33 |

Correlaciones de Pearson: **accuracy ~ `D_KL` = −0.659**, **accuracy ~ `Q` = 0.252**.
Con n=6 ninguna es distinguible de cero (p = 0.16 y 0.63).

Lo relevante no es la correlación sino las dos primeras filas. Para
`RealAmplitudes` con `reps=1`, los entrelazamientos `full` y `reverse_linear`
**implementan el mismo unitario con el mismo orden de parámetros** —el primero con
6 CNOT y el segundo con 3—, y por eso sus descriptores coinciden hasta el sexto
decimal. Sus accuracies son 0.93 y 0.33. Sesenta puntos de diferencia entre dos
modelos que admiten exactamente el mismo conjunto de funciones aprendibles: eso no
lo puede predecir ningún descriptor del circuito, y fija la escala del ruido de
entrenamiento frente al que hay que leer cualquier comparación de arquitecturas.

> **Aviso de reproducibilidad.** El notebook **sí tiene salidas ejecutadas**, pero
> su celda del barrido fue editada después de correr y, tal como está guardada, no
> es Python válido:
>
> ```python
> acc, m = train_eval(fm, (an, ent, reps)      # paréntesis sin cerrar
> ```
>
> La llamada tampoco respeta la firma `train_eval(fm_name, an_name,
> an_entanglement, reps)`. Los descriptores de la tabla coinciden dígito a dígito
> con los del notebook 01, lo que confirma que los circuitos evaluados son los que
> se dicen; pero el estudio **no se puede regenerar desde el repositorio** hasta
> reparar esa celda. Conviene además repetirlo sobre varias semillas: con una sola
> ejecución por fila, la varianza del entrenamiento domina el resultado.

## 2. Prototipo combinado: clasificación + reconstrucción (`Hybrid CQNN-VQAE.ipynb`)

Notebook exploratorio que replica el flujo clásico completo (clasificación de
ruido + *denoising*) sustituyendo parte del procesamiento por una QNN:

- **Clasificación de ruido mixto** con una HQNN (CNN + `EstimatorQNN` de 4
  qubits, `zz_feature_map` + `real_amplitudes`). Esta sección se dejó como
  borrador (celdas comentadas) a la espera de fijar observables y `reps`; se
  incluye a cambio una red **clásica equivalente** ya entrenada como referencia,
  que en 10 épocas llega a `Val Acc = 1.0000` (el notebook **no** evalúa sobre
  el conjunto de test).
- **Denoising con `ConvHQVAE`**: encoder/decoder convolucional + bloque
  cuántico (`EstimatorQNN` de 4 qubits, `pauli_feature_map` + `real_amplitudes`,
  6 observables) en el espacio latente, con `latent_dim=4`, truco de
  reparametrización y pérdida β-VAE (`BCE + β·KL`, β=0.5), sobre 300/100/100
  muestras. Evaluado con **MSE** (0.0659) y **PSNR** (11.81 dB). El cálculo de
  **SSIM** quedó pendiente por falta de la dependencia `scikit-image` en el
  entorno de ejecución.

  > La celda está configurada con `EPOCHS = 15`, pero el registro que conserva
  > el notebook solo llega a la época 2; las métricas finales sí corresponden a
  > un modelo entrenado.

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
| `test-hqvae/results/` | Barrido de 6 configuraciones de circuito (`HQVAE4_PFM_RA_linear_r1`, `HQVAE4_ZZFM_RA_linear_r1`, `HQVAE6_ZZFM_ESU2_linear_r1`, `HQVAE6_ZZFM_RA_full_r1`, `HQVAE6_ZZFM_RA_linear_r1`, `HQVAE6_ZZFM_RA_linear_r2`) + comparación de optimizadores (`cobyla`, `rmsprop`, `sgd`, `spsa`) sobre la configuración base, más `sweep_metrics.csv` con las métricas agregadas. |

Dos advertencias necesarias para leer estos artefactos:

- **El barrido de circuitos hay que citarlo desde `sweep_metrics.csv`, no desde
  `HQVAE6_ZZFM_RA_linear_r1.html`.** La comparación de optimizadores se lanzó
  después con `train_baseline=False`, y su iteración `adam` **sobrescribió** ese
  HTML: el fichero muestra 0.0427 / 14.0168 / 0.4973 y ya no incluye la columna
  del ConvVAE, mientras que la fila homónima del CSV —la del barrido de
  circuitos, con baseline— es 0.042294 / 14.060 / 0.503226.
- **`sgd` ignora `cfg.lr`**: `_make_optimizer` fija `lr=1e-4, momentum=0.9` para
  esa rama, así que su comparación con el resto de optimizadores no es a
  igualdad de tasa de aprendizaje.

COBYLA y SPSA no son entrenamientos de una sola fase: hacen primero un warm-up
completo con Adam y después 45 iteraciones sin gradiente **solo sobre los pesos
del ansatz**, minimizando la pérdida de validación con el resto de la red
congelada.

## Historial de esta rama

- **`first experiment finished`**: commit único que introduce todo lo anterior
  — módulo y notebooks de expresibilidad (E6), el prototipo combinado
  `Hybrid CQNN-VQAE.ipynb`, y el testbed `test-hqvae/` con su primer barrido
  de circuitos y optimizadores.

## Entorno

`requirements.txt` no fija versiones. Las ejecuciones registradas en los
notebooks 01–02 corresponden a **Qiskit 2.5.0 + qiskit-machine-learning 0.9.0**
sobre **Python 3.12** en Google Colab, con **torch 2.11 + torchvision 0.26**
para el notebook 03 y los testbeds. Para las métricas SSIM del testbed HQVAE se
necesita además `scikit-image` (ya listado en `requirements.txt`, pero ausente
en el entorno donde se ejecutó `Hybrid CQNN-VQAE.ipynb`).

> Nota: el `.venv` incluido apunta a un intérprete de otra máquina y no es
> reutilizable aquí; crea un entorno nuevo (p. ej. `py -3.12 -m venv .venv` y
> `pip install -r requirements.txt`) o usa tu kernel habitual.

## Salidas

Los notebooks 01–03 escriben con la convención estable `E6_*` en el directorio
que fije `OUTPUT_DIR` al inicio de cada uno.

- Tablas: `E6_circuit_descriptors.csv`, `E6_expressibility.csv`, `E6_entangling.csv`,
  `E6_paper_reference.csv`, `E6_metric_vs_accuracy.csv`.
- Figuras (300 dpi): `E6_expressibility_bars.png`, `E6_entangling_bars.png`,
  `E6_expressibility_vs_reps.png`, `E6_fidelity_histograms.png`,
  `E6_paper_reference.png`, `E6_expr_vs_accuracy.png`, `E6_ent_vs_accuracy.png`.

> **Estos ficheros no están versionados.** `OUTPUT_DIR` se resuelve fuera del
> repositorio (en la ejecución guardada, `/TFM-EXPERIMENTS/outputs/`), de modo
> que ningún CSV ni PNG de E6 forma parte de esta rama. Las cifras que la
> memoria cita se leen de las salidas de los propios notebooks, no de esos
> ficheros.

## Parámetros de ejecución

`N_SAMPLES` (muestras de fidelidad: 2000 en el notebook 01 y 3000 en el 02; el
paper usa ~5000) y, en el notebook 03, `EPOCHS` y `CONFIGS`. Súbelos para las
cifras finales.
