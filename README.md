# experiment-4 — Testbed de clasificación (HQNN) y reproducibilidad multi-semilla

Rama construida sobre `experiment-3` que separa definitivamente la
**clasificación** de la **reconstrucción** en un testbed dedicado
(`test-hcqnn/`), examina el papel de la capa de lectura de la HQNN, barre tres
optimizadores sobre cuatro semillas fijas y añade repeticiones multi-semilla
tanto para clasificación como para reconstrucción de cara a la memoria del TFM.

## 1. Nuevo testbed de clasificación — `test-hcqnn/`

Notebook dedicado (`Hybrid CNN-QNN.ipynb`, con una copia de trabajo
`Hybrid CNN-QNN copy.ipynb`) que sustituye a la sección de clasificación —
hasta entonces solo esbozada— del antiguo `Hybrid CQNN-VQAE.ipynb`
(**eliminado en esta rama**, ya superado por este testbed y por `test-hqvae/`).

Estructura del notebook:

1. **Imports y semilla** para reproducibilidad.
2. **Inspección de la capa de lectura**: con `fc_final = nn.Identity()`, los
   valores esperados de los observables de Pauli se usan directamente como
   logits de `CrossEntropyLoss`. El primer observable (`ZIII + IZII`) tiene
   rango teórico `[-2, 2]` mientras que los otros dos (`ZZII`, `IIZZ`) están
   acotados a `[-1, 1]`, y la hipótesis de partida era que esa asimetría
   sesgaría el modelo hacia la clase 0. **Las ejecuciones no la respaldan**
   (ver el aviso más abajo).
3. **Variante con capa de lectura entrenable**: se sustituye
   `fc_final = nn.Identity()` por `nn.Linear(len(qnn.observables), 3)`, que
   aprende una transformación afín (escala y sesgo por clase) sobre los valores
   esperados, dejando intactos el circuito, el pipeline de datos, la semilla y
   el protocolo de entrenamiento.
4. **Red clásica equivalente** como baseline: mismo extractor convolucional
   (`conv1`, `conv2`, `fc1`) y mismo cuello de botella de dimensión 4,
   sustituyendo el circuito variacional por una capa densa clásica
   `nn.Linear(4, 3)`, para aislar el efecto de la capa cuántica.
5. **Comparativa final** de las métricas de los tres modelos bajo idéntico
   protocolo, exportada también a HTML.

### Protocolo común a todas las ejecuciones

`CrossEntropyLoss`, `epochs = 10`, early stopping con `patience = 3` sobre la
pérdida de validación, `batch_size = 32` y restauración del mejor `state_dict`,
sobre las mismas 300/100/100 muestras de MNIST con ruido mixto. Circuito de
referencia: `ZZFeatureMap` (entrelazamiento `full`, 1 repetición) +
`RealAmplitudes` (entrelazamiento `reverse_linear`, 1 repetición) sobre 4
qubits, con los observables `ZIII + IZII`, `ZZII` e `IIZZ`.

### Aviso sobre el diagnóstico del sesgo de lectura

> La celda de diagnóstico evalúa una QNN **sin entrenar** sobre un lote de 32
> muestras e imprime, por observable, mínimo/máximo/media/desviación y el
> `bincount` de la predicción mayoritaria. Lo que se observa en los artefactos
> ejecutados **no es un sesgo hacia la clase 0**:
>
> - En `Hybrid CNN-QNN.ipynb` (estado actual del notebook): medias
>   `[-0.209, 0.537, -0.027]` y `bincount = [0, 32]` → las 32 muestras caen en la
>   **clase 1**.
> - En `..._adamw_..._44.html`: medias `[-0.556, -0.097, -0.134]` y
>   `bincount = [0, 21, 11]` → reparto entre las clases 1 y 2, **cero** en la 0.
>
> Es decir, la red sin entrenar degenera hacia una o dos clases —comportamiento
> esperable en cualquier red no entrenada— y la dirección de esa degeneración
> depende de la inicialización aleatoria, no del rango de los observables. La
> hipótesis del sesgo estructural queda **sin confirmar** y, en consecuencia,
> este diagnóstico **no se reporta como resultado** en la memoria del TFM.
>
> La variante con `fc_final = nn.Linear(...)` tampoco mejora al modelo con
> `Identity`. En la comparativa de `..._adamw_..._44.html`, bajo el mismo
> protocolo y la misma semilla: HQNN con `Identity` **0.94**, HQNN con `Linear`
> **0.32**, CNN clásica **0.92**. La configuración adoptada en la memoria es la
> de `Identity`.

## Resultados (`test-hcqnn/results/`)

Barrido de **feature map** (`PFM`=Pauli, `ZZFM`=ZZ) × **ansatz**
(`ESU`=EfficientSU2, `RA`=RealAmplitudes) × capa de lectura
(`fc_final=Identity` vs `fc_final=Linear`) × **optimizador (`adam`, `adamw`,
`rmsprop`)**, con semillas fijas `42`–`45`.

### Comparación de optimizadores sobre la configuración de referencia

Configuración `HCQNN4_ZZFM_full__r1_RA_rlinear_r1_*_fc_final=Identity`, con los
mismos observables en las tres familias:

| Semilla | HQNN Adam | HQNN AdamW | HQNN RMSprop | CNN AdamW | CNN RMSprop |
|---|---|---|---|---|---|
| 42 | 1.00 | 1.00 | 0.31 | — | 0.31 |
| 43 | 0.86 | 0.91 | 0.35 | 0.92 | 0.35 |
| 44 | 0.99 | 0.94 | 0.32 | 0.92 | 0.32 |
| 45 | 0.99 | 1.00 | 0.29 | 1.00 | 0.37 |

**RMSprop falla en los dos modelos por igual**: con soportes de test ≈31/31/38,
accuracies de 0.29–0.37 corresponden a un predictor prácticamente constante, y
HQNN y CNN coinciden dígito a dígito en tres de las cuatro semillas. El fallo,
por tanto, está en la optimización y no en la capa cuántica. Es el criterio por
el que la memoria adopta **AdamW**.

**Dos ejecuciones no sirven como baseline clásico.** En
`..._adam_..._44.html` y `..._adam_..._45.html` la línea activa es

```python
optimizer_classical = optim.Adam(hqnn.parameters(), lr=0.001)
```

es decir, el optimizador «clásico» recibe los parámetros de la **HQNN**, no los
de `classical_cnn`. La CNN nunca se entrena y las accuracies clásicas que esos
dos informes muestran (**0.30** y **0.37**) son las de una red sin entrenar. No
deben leerse como resultados. Las ejecuciones con `adamw` y con `rmsprop` sí
construyen el optimizador sobre `classical_cnn.parameters()` y son válidas.

Con eso, la **única comparación controlada CNN vs HQNN** disponible es la de
AdamW en las semillas 43/44/45: HQNN 0.91 / 0.94 / 1.00 frente a CNN
0.92 / 0.92 / 1.00. Los informes `adam_42`, `adam_43` y `adamw_42` no incluyen
baseline clásico.

**Coste**: la HQNN tarda ≈795–828 s por ejecución con Adam/AdamW —y 208–522 s
con RMSprop, que se detiene antes por early stopping— frente a 0.14–0.77 s de la
CNN equivalente en la misma máquina.

### Artefactos

- `HCQNN4_PFM_full__r1_ESU_rlinear_r1_adam_fc_final=Identity.html` y
  `..._fc_final=Linear(_2..._5).html`
- `HCQNN4_ZZFM_full__r1_RA_rlinear_r1_adam_fc_final=Identity(_2, _42..._45).html`
- `HCQNN4_ZZFM_full__r1_RA_rlinear_r1_adam_fc_final=Linear.html`
- `HCQNN4_ZZFM_full__r1_RA_rlinear_r1_adamw_fc_final=Identity_42..._45.html`
- `HCQNN4_ZZFM_full__r1_RA_rlinear_r1_rmsprop_fc_final=Identity_42..._45.html`
  *(añadidos en los commits `new experiments with rmsprop` y
  `rmsprop corrected`; los cuatro usan los mismos observables que las series
  `adam`/`adamw`, por lo que la tabla anterior es homogénea)*
- `RESULTADO QUE ORIGINA Hybrid_CNN_QNN_ampliado.html` — resultado de
  referencia que motivó ampliar este testbed a partir del prototipo original.

## 2. Reproducibilidad multi-semilla del HQVAE (`test-hqvae/`)

Sobre la configuración de 6 qubits con entrelazamiento completo en el feature
map (`ZZFM_full`) y ansatz RealAmplitudes lineal, se repite el entrenamiento
con **AdamW** para `latent_dim ∈ {6, 12}` y semillas `42`–`46`, para evaluar
la estabilidad de las métricas de reconstrucción entre ejecuciones:

- `HQVAE6_ZZFM_full_RA_linear_r1_adamw_12_42..._46.html`
- `HQVAE6_ZZFM_full_RA_linear_r1_adamw_6_42..._44.html`

Dos advertencias sobre esta serie, a diferencia de la de `experiment-1` a
`experiment-3`:

- Estos notebooks **no llaman a `hqvae_lib.py`**: reimplementan los modelos por
  su cuenta. Como consecuencia, sus informes **no incluyen los descriptores de
  circuito** ni calculan **SSIM** (solo MSE y PSNR).
- En `..._adamw_12_43.html` las dos últimas celdas de evaluación **no están
  ejecutadas** (`In [ ]`), y los valores de MSE/PSNR que el informe muestra
  (`0.034931`, `0.035528`) son literalmente los de la semilla 42, arrastrados
  del texto de la celda anterior. Esa ejecución debe excluirse de cualquier
  agregado.

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
- **`test bed de hybrid cnn-qnn.ipynb good experiments`**: añade el diagnóstico
  de la capa de lectura y la variante con `fc_final=Linear`; primeros
  resultados del barrido PFM/ZZFM × ESU/RA × Identity/Linear y su exportación
  a HTML.
- **`several tests cqnn and vqae`**: elimina el prototipo `Hybrid
  CQNN-VQAE.ipynb` (ya superado por `test-hcqnn/` y `test-hqvae/`); añade las
  repeticiones multi-semilla (42–45) de clasificación y (42–46) del HQVAE
  con AdamW a `latent_dim` 6 y 12.
- **`updated readme exp 4`**: documentación de la rama.
- **`new experiments with rmsprop`** y **`rmsprop corrected`**: añaden la
  tercera familia de optimizador (semillas 42–45), re-ejecutada para que use
  los mismos observables que las series `adam` y `adamw` y sea comparable con
  ellas.

## Entorno

`requirements.txt` no fija versiones. Los informes HTML de esta rama se
exportaron desde Google Colab; las ejecuciones registradas de los notebooks
`01`/`02` de expresibilidad corresponden a **Qiskit 2.5.0 +
qiskit-machine-learning 0.9.0** sobre **Python 3.12**, junto con
**torch + torchvision**.
