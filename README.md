# experiment-3 — Barrido de dimensión latente y optimizadores (HQVAE)

Rama construida sobre `experiment-2` que amplía la prueba dirigida de SPSA a
un **estudio sistemático** sobre el testbed `test-hqvae/`: cómo afecta la
**dimensión del espacio latente** y la **elección de optimizador** a la
calidad de reconstrucción (MSE/PSNR/SSIM) del HQVAE de 6 qubits.

## Qué aporta esta rama

- **Barrido de dimensión latente con SPSA**: se entrena la misma
  configuración de circuito (`HQVAE6_ZZFM_RA_linear_r1`, feature map ZZ +
  ansatz RealAmplitudes con entrelazamiento lineal) variando
  `latent_dim ∈ {4, 8, 12}`, manteniendo el optimizador SPSA. El objetivo es
  observar si desacoplar el espacio latente clásico del número de qubits
  (vía el adaptador `Linear(latent_dim, n_qubits)` de `hqvae_lib.py`) afecta
  a la fidelidad de la reconstrucción.
- **Comparación de optimizadores a `latent_dim=12`**: sobre esa misma
  configuración de circuito, se entrena con **Adam, AdamW, RMSprop y SPSA**
  para contrastar optimizadores basados en gradiente end-to-end frente al
  optimizador gradient-free (SPSA), que solo ajusta los pesos del ansatz
  cuántico.
- **Semilla actualizada a `SEED = 44`**, propagada igual que en
  `experiment-2` a la carga de datos y a cada `HQVAEConfig`.
- Se mantiene la celda de dispersión **expresibilidad/entrelazamiento (KL, Q)
  vs. calidad de reconstrucción (SSIM, PSNR)** del barrido `default_sweep`,
  heredada del testbed original, para relacionar los descriptores del
  experimento E6 con el rendimiento en la tarea de denoising.

## Contenido relevante de esta rama

| Fichero | Qué es |
|---|---|
| `test-hqvae/test_hqvae.ipynb` | Notebook driver, con `SEED=44` y el run único parametrizado para explorar `latent_dim` (4/8/12) manteniendo SPSA, además de la comparación de optimizadores. |
| `test-hqvae/results/HQVAE6_ZZFM_RA_linear_r1_spsa_4.html` | SPSA, `latent_dim=4`. |
| `test-hqvae/results/HQVAE6_ZZFM_RA_linear_r1_spsa_8.html` | SPSA, `latent_dim=8`. |
| `test-hqvae/results/HQVAE6_ZZFM_RA_linear_r1_spsa_12.html` | SPSA, `latent_dim=12`. |
| `test-hqvae/results/HQVAE6_ZZFM_RA_linear_r1_adam_12.html` | Adam, `latent_dim=12`. |
| `test-hqvae/results/HQVAE6_ZZFM_RA_linear_r1_adamw_12.html` | AdamW, `latent_dim=12`. |
| `test-hqvae/results/HQVAE6_ZZFM_RA_linear_r1_rmsprop_12.html` | RMSprop, `latent_dim=12`. |

El resto del contenido (E6, `Hybrid CQNN-VQAE.ipynb`, `hqvae_lib.py`) es el
mismo que en `experiment-1`/`experiment-2`.

## Historial de esta rama

- **`first experiment finished`** *(heredado)*: base común del proyecto.
- **`experiment-2 tested (spsa, seed 43)`** *(heredado)*: primera prueba
  dirigida de SPSA con semilla 43.
- **`removed before experiment html results`**: se elimina el resultado SPSA
  de `experiment-2` antes de lanzar la nueva tanda de pruebas de esta rama.
- **`tested different latent dims with spsa`**: añade los resultados SPSA
  para `latent_dim ∈ {4, 8, 12}` y actualiza el notebook driver.
- **`HQVAE6_ZZFM_RA_linear_r1_rmsprop_12`**: añade los resultados de Adam,
  AdamW y RMSprop a `latent_dim=12`, para compararlos con SPSA.
- **`HQVAE6_ZZFM_RA_linear_r1_spsa_12`**: recalcula/actualiza el resultado
  SPSA a `latent_dim=12` (semilla 44) y ajusta el notebook driver.

## Entorno

Igual que en las ramas anteriores: **Qiskit 2.3 + qiskit-machine-learning 0.9**
y **torch + torchvision**, según `requirements.txt`.
