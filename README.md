# experiment-2 — Prueba dirigida de SPSA (semilla 43)

Rama construida sobre `experiment-1` que se centra en una prueba concreta y
reproducible del testbed de reconstrucción `test-hqvae/`: entrenar el HQVAE
con el optimizador **SPSA** (gradient-free, actúa solo sobre los pesos del
ansatz cuántico) fijando una semilla explícita distinta a la del experimento
anterior, para comprobar reproducibilidad y estabilidad del entrenamiento.

## Qué cambia respecto a `experiment-1`

- **Semilla fija `SEED = 43`** propagada de forma consistente al notebook
  driver (`torch.manual_seed`, `np.random.seed`), a la carga de datos
  (`load_denoising_mnist(..., seed=SEED)`) y al barrido de optimizadores
  (`HQVAEConfig(..., seed=SEED)`), de modo que la carga de MNIST con ruido
  mixto y la inicialización de los modelos sean reproducibles.
- **Run único reconfigurado** a `n_qubits=6`, `feature_map='zz'`,
  `ansatz='real_amplitudes'` (entrelazamiento lineal), `latent_dim=8`,
  `epochs=15`, `optimizer='spsa'` — en lugar del optimizador por defecto
  (Adam) usado en `experiment-1`.
- **Limpieza de resultados**: se eliminaron del repositorio los HTML del
  primer barrido de `experiment-1` (6 configuraciones de circuito + 4
  optimizadores), el `sweep_metrics.csv` y el `test_hqvae.html` exportado, ya
  que correspondían a una ejecución anterior. Solo se conserva el resultado
  de esta prueba dirigida:
  `test-hqvae/results/HQVAE6_ZZFM_RA_linear_r1_spsa.html`.

El resto del contenido (módulo de expresibilidad E6, notebooks 01–03, el
prototipo combinado `Hybrid CQNN-VQAE.ipynb` y la librería `hqvae_lib.py`) se
mantiene igual que en `experiment-1`; ver el README de esa rama para la
descripción completa.

## Contenido relevante de esta rama

| Fichero | Qué es |
|---|---|
| `test-hqvae/test_hqvae.ipynb` | Notebook driver, ahora con `SEED=43` fijado y el run único apuntando a la configuración SPSA de 6 qubits. |
| `test-hqvae/results/HQVAE6_ZZFM_RA_linear_r1_spsa.html` | Resultado autocontenido de esa ejecución (config `HQVAE6_ZZFM_RA_linear_r1`, optimizador SPSA, semilla 43). |
| `test-hqvae/hqvae_lib.py` | Sin cambios respecto a `experiment-1`: código reutilizable del testbed (ruido mixto, `ConvVAE`, `ConvHQVAE`, entrenamiento, informes HTML). |
| `qexpr.py`, `01`–`03_*.ipynb` | Módulo y notebooks de expresibilidad/entrelazamiento (E6), sin cambios respecto a `experiment-1`. |

## Historial de esta rama

- **`first experiment finished`** *(heredado de `experiment-1`)*: base común
  del proyecto (E6 + testbed HQVAE + prototipo combinado).
- **`experiment-2 tested (spsa, seed 43)`**: fija la semilla 43 en todo el
  pipeline de `test-hqvae/`, reconfigura el run único del HQVAE de 6 qubits
  para usar el optimizador SPSA, y elimina los resultados HTML/CSV de la
  ejecución anterior que ya no correspondían a esta prueba.

## Entorno

Igual que en `experiment-1`: **Qiskit 2.3 + qiskit-machine-learning 0.9** y
**torch + torchvision**, según `requirements.txt`.
