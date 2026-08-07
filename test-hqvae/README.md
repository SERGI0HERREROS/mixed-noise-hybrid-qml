# test-hqvae — Testbed de reconstrucción con HQVAE

Sandbox para **lanzar pruebas** de reconstrucción (denoising) de imágenes de **MNIST con
ruido mixto** usando **HQVAE** (Hybrid Quantum Variational Autoencoder), comparándolo con un
**ConvVAE** clásico. Permite variar el circuito cuántico (nº de qubits, feature map, ansatz,
entrelazamiento, reps) y el optimizador, y guarda cada resultado como **HTML autocontenido**.

## Contenido

| Fichero | Qué es |
|---|---|
| `hqvae_lib.py` | Todo el código reutilizable: ruido mixto, dataset de denoising, `ConvVAE`, `ConvHQVAE`, entrenamiento, métricas, informes HTML y el barrido. |
| `test_hqvae.ipynb` | Notebook driver: setup → datos → run único → barrido (4 y 6 qubits) → comparación de optimizadores → export nbconvert. |
| `requirements.txt` | Dependencias (sin versiones fijadas). |
| `results/` | Se puebla al ejecutar: un `HQVAE<q>_<FM>_<ANSATZ>_<ent>_r<reps>.html` por run + `sweep_metrics.csv`. |

## Cómo ejecutar

No se crea ningún entorno virtual. Instala las dependencias en tu entorno (la primera celda
del notebook ya lo hace con `%%writefile` + `!pip install`), o manualmente:

```bash
pip install -r requirements.txt
```

Luego abre `test_hqvae.ipynb` y ejecuta las celdas de arriba abajo. El notebook usa
`../data` (MNIST ya descargado en `TFM/data`) y `../expressibility/qexpr.py` para los
descriptores de circuito.

## Base teórica

- **Expresibilidad y entrelazamiento** (Sim, Johnson & Aspuru-Guzik 2019, arXiv:1905.10876):
  cada circuito del barrido se anota con su expresibilidad `KL` (divergencia frente a Haar) y su
  `Q` de Meyer-Wallach, reutilizando el módulo `qexpr.py` del experimento E6. El barrido permite
  observar si mayor expresibilidad/entrelazamiento se traduce en mejor PSNR/SSIM (análogo al
  notebook `03_metric_vs_accuracy`, pero para reconstrucción en lugar de clasificación).
- **Ruido mixto** (paper *4-QBITS C-Q NN for mixed noise type ID in images*): 3 mezclas por pares
  (gaussiano+speckle, gaussiano+sal-y-pimienta, sal-y-pimienta+speckle).
- **Optimizador** (motivado por el paper *DE-VQE*): además de Adam se pueden usar otros
  (`adamw`, `sgd`, `rmsprop`, `lbfgs`) end-to-end y, de forma experimental, `cobyla`/`spsa`
  (gradient-free, solo sobre los pesos del ansatz). **No** se implementa evolución diferencial.

## Arquitectura HQVAE

`ConvHQVAE` = encoder convolucional → `μ`, `logσ²` (latente clásico de `latent_dim`) →
reparametrización → **adaptador `Linear(latent_dim, n_qubits)` + `tanh()·π`** (encoding angular)
→ **QNN** (`EstimatorQNN`: feature map + ansatz, observables Z/ZZ) → se **concatena** la salida
del QNN con el latente `z` → decoder convolucional (`Sigmoid`). Pérdida β-VAE = `BCE + β·KL`.

El adaptador de ángulos **desacopla `latent_dim` de `n_qubits`**, así que se pueden probar
libremente 4 y 6 qubits (o más) con cualquier `latent_dim`.

## Barrido por defecto (`default_sweep`)

| tag | qubits | feature_map | ansatz | entanglement | reps |
|---|---|---|---|---|---|
| `HQVAE6_ZZFM_RA_linear_r1` | 6 | zz (full) | real_amplitudes | linear | 1 |
| `HQVAE4_ZZFM_RA_linear_r1` | 4 | zz (full) | real_amplitudes | linear | 1 |
| `HQVAE6_ZZFM_RA_full_r1` | 6 | zz (full) | real_amplitudes | full | 1 |
| `HQVAE6_ZZFM_ESU2_linear_r1` | 6 | zz (full) | efficient_su2 | linear | 1 |
| `HQVAE4_PFM_RA_linear_r1` | 4 | pauli [X,Y,ZZ] | real_amplitudes | linear | 1 |
| `HQVAE6_ZZFM_RA_linear_r2` | 6 | zz (full) | real_amplitudes | linear | 2 |

## Notas

- La capa cuántica usa `StatevectorEstimator` (simulación exacta en CPU), así que el coste
  crece con el nº de qubits y el tamaño del subset. Con 300/100/100 y 4–6 qubits es viable;
  baja `epochs` para pruebas rápidas.
- Convención de nombres HTML: `HQVAE<qubits>_<ZZFM|PFM>_<RA|ESU2>_<entanglement>_r<reps>`
  (+ sufijo `_<optimizador>` si no es `adam`), coherente con los HTML de `RESULTS/`.
