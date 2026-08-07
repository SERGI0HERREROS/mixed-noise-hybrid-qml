# IMPORTS NECESARIOS
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import torch



def plot_training_curves(train_loss, val_loss, train_acc, val_acc):
    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title("Loss curve")
    plt.show()

    plt.figure()
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Val Accuracy")
    plt.legend()
    plt.title("Accuracy curve")
    plt.show()

def plot_confusion_matrix(cm, classes, normalize=True, title="Confusion Matrix"):
    """
    Matriz de confusión mejorada para clasificación multiclase (3 clases o más)
    """

    cm = np.array(cm)

    # Normalización (MUY importante para TFM)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # evita NaN si hay filas vacías

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")

    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Anotaciones dentro de cada celda
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]

            if normalize:
                text = f"{value:.2f}"
            else:
                text = f"{int(value)}"

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if value > thresh else "black"
            )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds)

    return {"accuracy": acc, "confusion_matrix": cm, "report": report}


def mostrar_reconstrucciones(model, dataset, n=8, device="cpu"):
    model.eval()
    indices = np.random.choice(len(dataset), n)
    imgs_noisy = []
    imgs_clean = []

    for idx in indices:
        noisy, clean = dataset[idx]
        imgs_noisy.append(noisy)
        imgs_clean.append(clean)

    x_noisy = torch.stack(imgs_noisy).to(device)
    with torch.no_grad():
        recon, _, _ = model(x_noisy)

    x_noisy = x_noisy.cpu().view(-1, 28, 28)
    recon = recon.cpu().view(-1, 28, 28)
    imgs_clean = torch.stack(imgs_clean).view(-1, 28, 28)

    plt.figure(figsize=(12, 4))
    for i in range(n):
        plt.subplot(3, n, i+1)
        plt.imshow(imgs_clean[i], cmap='gray')
        plt.axis('off')
        plt.title("Original")

        plt.subplot(3, n, n+i+1)
        plt.imshow(x_noisy[i], cmap='gray')
        plt.axis('off')
        plt.title("Ruidosa")

        plt.subplot(3, n, 2*n+i+1)
        plt.imshow(recon[i], cmap='gray')
        plt.axis('off')
        plt.title("Reconstr.")
    plt.show()

