# Quanvolutional_NN_classify_reconstruct

TFM sobre clasificación de ruido mixto y *denoising* de imágenes con modelos
híbridos cuántico-clásicos (CNN + QNN variacional vía Qiskit/`EstimatorQNN`),
comparados frente a sus contrapartidas puramente clásicas (CNN, ConvVAE).

## Estado de esta rama

`develop` es la rama base/de integración del repositorio: **no contiene el
código experimental**, solo los ficheros comunes a todo el proyecto
(dependencias, configuración de entorno y este README). El trabajo
experimental real —notebooks, módulos de circuitos cuánticos, barridos de
hiperparámetros y resultados— vive en las ramas `experiment-1` a
`experiment-5`, cada una construida incrementalmente sobre la anterior.

## Ramas experimentales

| Rama | Contenido |
|---|---|
| [`experiment-1`](../../tree/experiment-1) | Primer experimento: métricas de expresibilidad/entrelazamiento de circuitos (E6) + testbed inicial de reconstrucción híbrida (HQVAE) con barrido de circuitos y optimizadores. |
| [`experiment-2`](../../tree/experiment-2) | Prueba dirigida del optimizador SPSA (gradient-free) sobre el HQVAE, con semilla alternativa. |
| [`experiment-3`](../../tree/experiment-3) | Barrido sistemático de dimensión latente (4/8/12) con SPSA y comparación de optimizadores (Adam, AdamW, RMSprop, SPSA) sobre el HQVAE de 6 qubits. |
| [`experiment-4`](../../tree/experiment-4) | Testbed dedicado del clasificador híbrido (HQNN): diagnóstico y corrección de un sesgo en la capa de lectura, comparación con baseline clásico, y repeticiones multi-semilla (42-46) de clasificación y reconstrucción. |
| [`experiment-5`](../../tree/experiment-5) | Análisis de explicabilidad (XAI: Saliency, Integrated Gradients, Grad-CAM, Occlusion, LIME, SHAP) comparando el modelo híbrido frente a su contrapartida clásica, tanto en clasificación como en reconstrucción. |

## Historial de esta rama

- **`first experiment finished`**: commit inicial compartido con el resto de
  ramas, con todo el código del primer experimento (E6 + HQVAE).
- **`remove files from dev branch`**: se eliminó de `develop` el código
  experimental (notebooks, módulos, resultados), dejando únicamente los
  ficheros de configuración comunes, para que esta rama actúe como base
  limpia de la que parten las ramas `experiment-*`.

## Entorno

Dependencias comunes en `requirements.txt` (Qiskit 2.x, qiskit-machine-learning,
numpy/pandas/matplotlib, scikit-learn, scikit-image, pylatexenc). PyTorch se
instala aparte según el hardware disponible (CPU o CUDA), como se indica en el
propio fichero.