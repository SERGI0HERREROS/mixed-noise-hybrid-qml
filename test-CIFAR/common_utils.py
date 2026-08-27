"""
common_utils.py
----------------
Utilidades compartidas por los notebooks CIFAR-100 de clasificación de ruido mixto
(CNN clásica / HQNN híbrida) y denoising (ConvVAE clásico / ConvHQVAE híbrido)
del TFM.

NOTA IMPORTANTE:
El notebook de referencia (Classical_CNN_VAE.ipynb) importa estas funciones
desde un módulo `common_utils` que no formaba parte de los ficheros
proporcionados. Este archivo reconstruye una implementación funcionalmente
equivalente a partir de cómo se invocan dichas funciones en el notebook de
referencia (mismas firmas, mismos nombres de argumentos, mismo tipo de
retorno). Si dispones del `common_utils.py` original del proyecto, puedes
sustituir este fichero por el tuyo: ambos notebooks generados únicamente
requieren que existan `plot_training_curves`, `plot_confusion_matrix`,
`evaluate_model` y `mostrar_reconstrucciones` con estas firmas.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def plot_training_curves(train_loss_list, val_loss_list, train_acc_list, val_acc_list):
    """Curvas de pérdida y accuracy (train/val) a lo largo de las épocas."""
    epochs = range(1, len(train_loss_list) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_loss_list, marker="o", label="Train")
    axes[0].plot(epochs, val_loss_list, marker="o", label="Val")
    axes[0].set_title("Curva de pérdida")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc_list, marker="o", label="Train")
    axes[1].plot(epochs, val_acc_list, marker="o", label="Val")
    axes[1].set_title("Curva de accuracy")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def evaluate_model(model, loader, device):
    """Evalúa un clasificador sobre `loader` y devuelve accuracy, informe de
    clasificación y matriz de confusión."""
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    target_names = ["gaussian+speckle", "gaussian+saltpepper", "saltpepper+speckle"]

    return {
        "accuracy": accuracy_score(all_targets, all_preds),
        "report": classification_report(all_targets, all_preds, target_names=target_names),
        "confusion_matrix": confusion_matrix(all_targets, all_preds),
        "y_true": all_targets,
        "y_pred": all_preds,
    }


def plot_confusion_matrix(cm, labels):
    """Heatmap simple de una matriz de confusión ya calculada."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Clase real")
    ax.set_title("Matriz de confusión")

    thresh = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def mostrar_reconstrucciones(model, dataset, n=8, device="cpu"):
    """Muestra en una cuadrícula: entrada ruidosa / imagen limpia / reconstrucción,
    para `n` ejemplos de `dataset` (que debe devolver pares (noisy, clean)).
    Funciona tanto con modelos que exponen `.reconstruct(x)` (uso de mu
    determinista) como con modelos cuyo `forward` devuelve (recon, mu, logvar).
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=n, shuffle=False)
    noisy, clean = next(iter(loader))
    noisy, clean = noisy.to(device), clean.to(device)

    with torch.no_grad():
        if hasattr(model, "reconstruct"):
            recon = model.reconstruct(noisy)
        else:
            out = model(noisy)
            recon = out[0] if isinstance(out, tuple) else out

    noisy_np = noisy.cpu().numpy()
    clean_np = clean.cpu().numpy()
    recon_np = recon.cpu().numpy()

    n = min(n, noisy_np.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))

    def _disp(a):
        """CHW -> HW (1 canal) o HWC (3 canales), listo para imshow."""
        a = np.squeeze(a)
        if a.ndim == 3 and a.shape[0] in (1, 3):
            a = np.transpose(a, (1, 2, 0))
        return np.clip(a, 0, 1)

    cmap = "gray" if noisy_np.shape[1] == 1 else None

    for i in range(n):
        axes[0, i].imshow(_disp(noisy_np[i]), cmap=cmap)
        axes[0, i].axis("off")

        axes[1, i].imshow(_disp(clean_np[i]), cmap=cmap)
        axes[1, i].axis("off")

        axes[2, i].imshow(_disp(recon_np[i]), cmap=cmap)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Noisy", fontsize=10)
    axes[1, 0].set_ylabel("Clean", fontsize=10)
    axes[2, 0].set_ylabel("Recon", fontsize=10)
    plt.suptitle("Reconstrucciones: entrada ruidosa / imagen limpia / reconstrucción")
    plt.tight_layout()
    plt.show()
