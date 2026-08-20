# experiment-4 — Testbed de clasificación (HQNN) y reproducibilidad multi-semilla

Rama construida sobre `experiment-3` que separa definitivamente la
**clasificación** de la **reconstrucción** en un testbed dedicado
(`test-hcqnn/`), diagnostica y corrige un sesgo en la capa de lectura de la
HQNN, y añade repeticiones multi-semilla tanto para clasificación como para
reconstrucción de cara a la memoria del TFM.

## 1. Nuevo testbed de clasificación — `test-hcqnn/`

Notebook dedicado (`Hybrid CNN-QNN.ipynb`, con una copia de trabajo
`Hybrid CNN-QNN copy.ipynb`) que sustituye a la sección de clasificación —
hasta entonces solo esbozada— del antiguo `Hybrid CQNN-VQAE.ipynb`
(**eliminado en esta rama**, ya superado por este testbed y por `test-hqvae/`).

Estructura del notebook:

1. **Imports y semilla** para reproducibilidad.
2. **Diagnóstico del sesgo de la QNN**: con `fc_final = nn.Identity()`, los
   valores esperados de los observables de Pauli se usan directamente como
   logits de `CrossEntropyLoss`. El primer observable (`ZIII + IZII`) tiene
   rango teórico `[-2, 2]` mientras que los otros dos (`ZZII`, `IIZZ`) están
   acotados a `[-1, 1]`; esa asimetría puede sesgar el modelo hacia la clase 0
   independientemente de la entrada. Se comprueba empíricamente con una QNN
   sin entrenar.
3. **HQNN corregida**: se sustituye `fc_final = nn.Identity()` por una capa
   entrenable `nn.Linear(len(qnn.observables), 3)` que recalibra (escala y
   sesgo por clase) los valores esperados antes de la softmax implícita en
   `CrossEntropyLoss`, manteniendo el resto de la arquitectura, el pipeline
   de datos, la semilla y el protocolo de entrenamiento sin cambios.
4. **Red clásica equivalente** como baseline: mismo extractor convolucional
   (`conv1`, `conv2`, `fc1`) y mismo cuello de botella de dimensión 4,
   sustituyendo el circuito variacional por una capa densa clásica
   `nn.Linear(4, 3)`, para aislar el efecto de la capa cuántica.
5. **Comparativa final** de las métricas de los tres modelos (HQNN original
   con `fc_final=Identity`, HQNN corregida con `fc_final=Linear`, y baseline
   clásico) bajo idéntico protocolo, exportada también a HTML.

## Resultados (`test-hcqnn/results/`)

Barrido de combinaciones de **feature map** (`PFM`=Pauli, `ZZFM`=ZZ) ×
**ansatz** (`ESU`=EfficientSU2, `RA`=RealAmplitudes) × capa de lectura
(`fc_final=Identity` vs `fc_final=Linear`) × optimizador (`adam`, `adamw`),
con varias repeticiones y semillas fijas (`42`–`45`) para comprobar la
estabilidad de la corrección del sesgo:

- `HCQNN4_PFM_full__r1_ESU_rlinear_r1_adam_fc_final=Identity/Linear(_2..._5).html`
- `HCQNN4_ZZFM_full__r1_RA_rlinear_r1_adam_fc_final=Identity(_2, _42..._45).html`
- `HCQNN4_ZZFM_full__r1_RA_rlinear_r1_adam_fc_final=Linear.html`
- `HCQNN4_ZZFM_full__r1_RA_rlinear_r1_adamw_fc_final=Identity_42..._45.html`
- `RESULTADO QUE ORIGINA Hybrid_CNN_QNN_ampliado.html` — resultado de
  referencia que motivó ampliar este testbed a partir del prototipo original.

## 2. Reproducibilidad multi-semilla del HQVAE (`test-hqvae/`)

Sobre la configuración de 6 qubits con entrelazamiento completo en el feature
map (`ZZFM_full`) y ansatz RealAmplitudes lineal, se repite el entrenamiento
con **AdamW** para `latent_dim ∈ {6, 12}` y semillas `42`–`46`, para evaluar
la estabilidad de las métricas de reconstrucción (MSE/PSNR/SSIM) entre
ejecuciones:

- `HQVAE6_ZZFM_full_RA_linear_r1_adamw_12_42..._46.html`
- `HQVAE6_ZZFM_full_RA_linear_r1_adamw_6_42..._44.html`

Se conservan además todos los resultados de `experiment-3` (barrido de
`latent_dim` y optimizadores con SPSA/Adam/AdamW/RMSprop a `latent_dim=12`).

## Historial de esta rama

- **`first experiment finished`**, **`experiment-2 tested (spsa, seed 43)`**,
  **`removed before experiment html results`**, **`tested different latent
  dims with spsa`**, **`HQVAE6_ZZFM_RA_linear_r1_rmsprop_12`**,
  **`HQVAE6_ZZFM_RA_linear_r1_spsa_12`** *(heredados)*: base común y estudio
  de `latent_dim`/optimizadores del HQVAE (ver READMEs de `experiment-1` a
  `experiment-3`).
- **`testbed de hybrid cnn-qnn copy.ipynb`**: añade el nuevo testbed de
  clasificación `test-hcqnn/` con los notebooks `Hybrid CNN-QNN.ipynb` y
  `Hybrid CNN-QNN copy.ipynb`.
- **`test bed de hybrid cnn-qnn.ipynb good experiments`**: desarrolla el
  diagnóstico del sesgo de la QNN y la corrección con `fc_final=Linear`;
  añade los primeros resultados del barrido PFM/ZZFM × ESU/RA ×
  Identity/Linear y su exportación a HTML.
- **`several tests cqnn and vqae`**: elimina el prototipo `Hybrid
  CQNN-VQAE.ipynb` (ya superado por `test-hcqnn/` y `test-hqvae/`); añade las
  repeticiones multi-semilla (42–45) de la HQNN corregida y (42–46) del HQVAE
  con AdamW a `latent_dim` 6 y 12.

## Entorno

Igual que en las ramas anteriores: **Qiskit 2.3 + qiskit-machine-learning 0.9**
y **torch + torchvision**, según `requirements.txt`.
