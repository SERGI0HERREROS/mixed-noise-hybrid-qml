# experiment-6 — Réplica sobre CIFAR-100: control de validez de los resultados de MNIST

Rama construida sobre `experiment-5` que replica sobre **CIFAR-100** los
experimentos de reconstrucción y de explicabilidad que hasta ahora solo existían
para MNIST. El objetivo no es «probar con otro dataset», sino disponer de un
**control de validez** de las conclusiones obtenidas en las ramas anteriores.

## Por qué CIFAR-100 no es simplemente «MNIST más difícil»

En MNIST cerca del 80 % de los píxeles son fondo negro exacto (`x = 0`). Sobre
ese fondo, cada proceso de ruido se comporta de forma muy distinta:

- el **speckle**, definido como `x + x·η`, vale exactamente `0` cuando `x = 0`,
  es decir, **no existe** sobre el fondo;
- la **sal y pimienta** deja puntos blancos muy visibles sobre negro;
- el **gaussiano** sí ensucia el fondo entero.

La consecuencia es que la clase 2 (sal y pimienta + speckle), la única sin
componente gaussiana, conserva un fondo negro limpio con puntos blancos. Un CNN
puede identificarla mirando únicamente si el fondo está sucio, **sin aprender
nada sobre las estadísticas del ruido**.

En imágenes naturales ese atajo desaparece: no hay fondo a cero, el speckle
actúa sobre todos los píxeles y las tres clases se solapan de verdad. Si las
accuracies cercanas a 1.0 obtenidas en MNIST se mantienen en CIFAR-100, el
resultado gana solidez; si caen, significa que parte de aquel rendimiento estaba
midiendo un artefacto del dataset y no la capacidad del modelo.

## Contenido — carpeta `test-CIFAR/`

| Fichero | Equivalente en MNIST | Contenido |
|---|---|---|
| `Classical_CNN_CVAE.ipynb` | `XAI/Classical_CNN_CVAE.ipynb` | Rama clásica: CNN para clasificación + ConvVAE para denoising, con la batería completa de XAI. |
| `HQ_CNN_CVAE.ipynb` | `XAI/HQ_CNN_CVAE.ipynb` | Rama híbrida: HQNN (4 qubits) + ConvHQVAE (6 qubits en el latente), misma batería de XAI más la proyección post-QNN. |
| `vcae_VS_hqvae_gaussian_speckle.ipynb` | `test-hqvae/` homónimo | ConvVAE vs ConvHQVAE con ruido fijo gaussiano + speckle. |
| `vcae_VS_hqvae_gaussian_saltpepper.ipynb` | `test-hqvae/` homónimo | Ídem con gaussiano + sal y pimienta. |
| `vcae_VS_hqvae_saltpepper_speckle.ipynb` | `test-hqvae/` homónimo | Ídem con sal y pimienta + speckle. |
| `common_utils.py` | `XAI/common_utils.py` | Mismas utilidades, con `mostrar_reconstrucciones` adaptado a 1 y 3 canales. |

Los dos notebooks de XAI siguen siendo **gemelos** en el mismo sentido que en
`experiment-5`: comparten pipeline de datos, semilla (`SEED = 42`), particiones
300/100/100, arquitectura convolucional e hiperparámetros, y difieren
únicamente en el bloque cuántico. Cualquier diferencia observada entre ellos es
por tanto atribuible al componente cuántico y no a decisiones de diseño
colaterales.

## Configuración del dataset

Los cinco notebooks empiezan con el mismo bloque de configuración, y todo lo
demás se deriva de esas dos constantes:

```python
IMG_SIZE = 32     # 32 = resolución nativa de CIFAR-100
                  # 28 = redimensionado; todas las dimensiones internas
                  #      coinciden exactamente con los notebooks de MNIST
GRAYSCALE = True  # True  = 1 canal (recomendado)
                  # False = 3 canales RGB
```

| Variante | Canales | Mapa conv | Flatten VAE | Flatten clasificador |
|---|---|---|---|---|
| `IMG_SIZE = 32` | 1 o 3 | 8×8 | 4096 | 2048 |
| `IMG_SIZE = 28` | 1 o 3 | 7×7 | **3136** | **1568** |

Con `IMG_SIZE = 28, GRAYSCALE = True` las dimensiones internas son idénticas a
las de MNIST, de modo que la **única** variable que cambia entre ambos datasets
es el contenido de la imagen. Es la configuración más limpia para una
comparación controlada; la de 32×32 conserva la resolución nativa de CIFAR.

## Diferencias respecto a los notebooks de MNIST

Salvo las derivadas de la configuración anterior, se ha mantenido todo: misma
semilla, mismo split, mismos circuitos y observables, mismos optimizadores,
mismas épocas y mismo `beta = 0.5`. Los cambios son:

1. **Cargador**: `datasets.CIFAR100` con `Grayscale` y `Resize` opcionales. El
   submuestreo usa `Subset` en lugar de recortar `.data` / `.targets`, porque en
   CIFAR-100 `.targets` es una lista y no un tensor.
2. **Máscara de sal y pimienta compartida entre canales.** La versión de MNIST
   genera la máscara con `torch.rand_like(img)`, que en RGB produciría una
   máscara independiente por canal y, por tanto, píxeles de color aleatorio en
   lugar de impulsos blancos y negros. Aquí se genera con forma `(1, H, W)` y se
   difunde. Con 1 canal el comportamiento es idéntico al original.
3. **Visualización**: helper `to_disp()` que devuelve `HxW` o `HxWx3` según el
   número de canales, de modo que las mismas celdas sirven para ambas variantes.
4. **SSIM en los `vcae_VS_hqvae_*`**: los notebooks de MNIST solo calculaban MSE
   y PSNR. Aquí se añade SSIM para que las tres métricas sean directamente
   comparables con las tablas del capítulo de resultados de la memoria.
5. **Atribuciones multicanal**: en RGB, Saliency e Integrated Gradients se
   agregan sobre los canales para producir un mapa 2D comparable con el de la
   variante en escala de grises, y la ventana de Occlusion cubre todos los
   canales a la vez.

## Etiquetas: qué se está clasificando

Conviene subrayarlo porque se presta a confusión: las **100 categorías de
CIFAR-100 no se utilizan**. La etiqueta de este problema sigue siendo el tipo de
ruido mixto (tres clases), igual que en MNIST. CIFAR-100 aporta únicamente
imágenes con estadísticas más complejas; no se trata de «clasificación en
CIFAR-100» sino de clasificación de ruido **sobre imágenes de** CIFAR-100.

## Estado de esta rama

Los cinco notebooks están **implementados y verificados estructuralmente**, pero
**todavía no ejecutados**: no hay por tanto resultados ni exports HTML en esta
rama. La verificación realizada cubre:

- las 203 celdas de código parsean correctamente;
- `ConvVAE`, `ConvHQVAE`, `ClassicalNet` y `Net` hacen un forward pass válido en
  las tres variantes (32/gris, 28/gris, 32/RGB);
- los gradientes de los pesos del ansatz atraviesan el `TorchConnector` con
  valores finitos y no nulos;
- los circuitos construidos coinciden con los de MNIST (6 entradas / 12 pesos /
  6 observables en el VAE; 4 entradas / 3 observables en el clasificador);
- con `IMG_SIZE = 28` el aplanado da 3136 y 1568, exactamente los valores de los
  notebooks de MNIST.

## Ejecución

CIFAR-100 se descarga automáticamente a `./data` la primera vez (unos 169 MB).
El entorno es el mismo que en las ramas anteriores (**Qiskit 2.3 +
qiskit-machine-learning 0.9**, **torch + torchvision**), más `captum`, `lime`,
`shap` y `scikit-image` para los notebooks de explicabilidad. Estas cuatro
últimas no figuran en `requirements.txt` y hay que instalarlas aparte.

Cada notebook termina con una celda que exporta el resultado a HTML mediante
`nbconvert`, siguiendo la convención del resto del repositorio.

> **Aviso de coste.** Los modelos híbridos simulan el circuito en cada pasada
> forward. En MNIST el `ConvHQVAE` tardaba del orden de 85 minutos por ejecución
> con 300 imágenes; en CIFAR-100 a 32×32 el coste del bloque cuántico es el
> mismo (el latente no cambia), pero el encoder y el decoder procesan un 30 % más
> de píxeles. Conviene lanzar primero un solo `vcae_VS_hqvae_*` para medir el
> tiempo real antes de encadenar los cinco.

## Historial de esta rama

- **Commits heredados de `experiment-1` a `experiment-5`**: base común del
  proyecto, testbed de expresibilidad E6, testbed de reconstrucción
  `test-hqvae/` (barrido de circuitos, optimizadores y dimensión latente),
  testbed de clasificación `test-hcqnn/` (diagnóstico y corrección del sesgo de
  la capa de lectura, reproducibilidad multi-semilla) y estudio de
  explicabilidad `XAI/`. Ver los READMEs de esas ramas para el detalle completo.
- **Réplica sobre CIFAR-100**: añade la carpeta `test-CIFAR/` con los cinco
  notebooks descritos arriba y su `common_utils.py`.
