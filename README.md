# experiment-5 — Explicabilidad (XAI): clásico vs. híbrido cuántico-clásico

Rama construida sobre `experiment-4` que añade un estudio de **explicabilidad
(XAI)** comparando, celda a celda, la rama puramente clásica (CNN + ConvVAE)
frente a la rama híbrida cuántico-clásica (HQNN + ConvHQVAE), usando como
referencia las mejores configuraciones ya validadas en `experiment-4`
(`HCQNN4_ZZFM_full__r1_RA_rlinear_r1_adamw_fc_final_Identity_42` para
clasificación y `HQVAE6_ZZFM_full_RA_linear_r1_adamw_12_42` para
reconstrucción).

## Contenido — carpeta `XAI/`

Dos notebooks **gemelos**, con el mismo pipeline de datos, semilla
(`SEED=42`), particiones train/val/test, ruido mixto, arquitectura
convolucional e hiperparámetros de entrenamiento, para que solo difiera el
bloque cuántico:

| Notebook | Rama | Contenido |
|---|---|---|
| `Classical_CNN_CVAE.ipynb` | Clásica | CNN para clasificación + ConvVAE para denoising, ambos puramente clásicos. |
| `HQ_CNN_CVAE.ipynb` | Híbrida | HQNN (CNN + `EstimatorQNN` de 4 qubits, `zz_feature_map`+`real_amplitudes`) para clasificación + `ConvHQVAE` (bloque cuántico de 6 qubits en el espacio latente) para denoising. |

Cada notebook sigue la misma estructura en 6 secciones:

1. Preprocesamiento de datos y modelización del ruido mixto (idéntico en ambos).
2. Clasificación de ruido (CNN vs. HQNN).
3. Denoising (ConvVAE vs. ConvHQVAE).
4. **XAI del clasificador**: activaciones internas, Saliency Maps, Integrated
   Gradients, Grad-CAM, Occlusion Sensitivity, proyección t-SNE del espacio
   latente pre-bloque final (y también post-QNN en la rama híbrida), análisis
   de errores, LIME y SHAP (`GradientExplainer`).
5. **XAI del modelo de denoising**: reconstrucciones cualitativas, PCA/t-SNE
   del espacio latente `μ`, saliency de la reconstrucción, mapa de error
   residual, y *traversals* latentes (qué "imagina" el decoder al recorrer
   cada dimensión).
6. Conclusiones y guía explícita de qué comparar entre ambos notebooks
   (curvas de entrenamiento, separabilidad del espacio latente, coste
   computacional relativo de cada técnica XAI, geometría de `μ`,
   interpretabilidad de los traversals).

Nota metodológica destacada en el notebook híbrido: los métodos basados en
gradiente (Saliency, Integrated Gradients, Grad-CAM, SHAP) retropropagan sin
problema a través de la QNN gracias a `input_gradients=True` de
`EstimatorQNN`; los métodos basados en perturbación (Occlusion, LIME) no
dependen de diferenciabilidad, pero sí de una simulación completa del
circuito por cada perturbación evaluada, por lo que su coste computacional es
notablemente mayor que en la rama clásica.

## Resultados (`XAI/results/` y exports HTML)

| Fichero | Qué es |
|---|---|
| `results/Classical_CNN_CVAE_42.html` | Export completo del notebook clásico (semilla 42). |
| `results/HQ_CNN_CVAE_42.html` | Export completo del notebook híbrido (semilla 42). |
| `results/HQ_CNN_CVAElr04 42.html` | Variante del notebook híbrido con *learning rate* distinto (semilla 42). |
| `HQ_CNN_CVAE.html`, `HQ_CNN_CVAE_full.html`, `HQ_CNN_CVAE_reps=2.html` | Exports adicionales del notebook híbrido explorando variantes de configuración (entrelazamiento `full` y `reps=2` en el ansatz). |
| `common_utils.py` | Utilidades compartidas por ambos notebooks (curvas de entrenamiento, matriz de confusión, evaluación, reconstrucciones), copia local de la usada en el resto del proyecto. |

## Referencia de resultados obtenidos

Según la ejecución de referencia documentada en los propios notebooks
(semilla 42): la HQNN híbrida alcanzó accuracy de test perfecta (1.0000) en
clasificación; el `ConvHQVAE` obtuvo MSE≈0.0349 y PSNR≈14.99 dB en
denoising, muy próximos a los del `ConvVAE` clásico (MSE≈0.0355,
PSNR≈14.99 dB) entrenado con el mismo protocolo — evidencia de que, en este
régimen de pocos datos (300 muestras, 15 épocas), el bloque cuántico del
espacio latente ni penaliza ni mejora sustancialmente la fidelidad de
reconstrucción frente a su contrapartida clásica.

## Historial de esta rama

- **Commits heredados de `experiment-1` a `experiment-4`**: base común del
  proyecto, testbed de expresibilidad E6, testbed de reconstrucción
  `test-hqvae/` (barrido de circuitos, optimizadores y dimensión latente) y
  testbed de clasificación `test-hcqnn/` (diagnóstico y corrección del sesgo
  de la QNN, reproducibilidad multi-semilla). Ver los READMEs de esas ramas
  para el detalle completo.
- **`explainability branch`**: añade la carpeta `XAI/` completa —los
  notebooks gemelos `Classical_CNN_CVAE.ipynb` y `HQ_CNN_CVAE.ipynb`, sus
  exports HTML y `common_utils.py`— implementando la batería de técnicas de
  explicabilidad descrita arriba sobre las configuraciones de referencia
  validadas en `experiment-4`.

## Entorno

Igual que en las ramas anteriores (**Qiskit 2.3 + qiskit-machine-learning
0.9**, **torch + torchvision**, según `requirements.txt`), más las librerías
de explicabilidad usadas en `XAI/` (Captum para Saliency/Integrated
Gradients/Grad-CAM, `lime` y `shap`) y `scikit-learn`/`scikit-image` para las
proyecciones t-SNE/PCA y las métricas de reconstrucción.
