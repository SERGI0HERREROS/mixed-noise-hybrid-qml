"""hqvae_lib.py — Testbed de reconstrucción (denoising) de MNIST con ruido mixto
mediante HQVAE (Hybrid Quantum Variational Autoencoder), para el TFM de QML.

Todo el código reutilizable del sandbox `test-hqvae/` vive aquí:

  * Ruido mixto (3 clases) y `DenoisingMNISTDataset` -> (noisy, clean).
  * `HQVAEConfig`: configuración de un experimento (nº qubits, feature map, ansatz,
    entanglement, reps, latent_dim, beta, epochs, optimizador...).
  * `ConvVAE` (baseline clásico) y `ConvHQVAE` (híbrido con capa cuántica en el latente).
  * `train_vae`: entrenamiento end-to-end con optimizadores de PyTorch
    (adam/adamw/sgd/rmsprop/lbfgs) y, opcionalmente, refinado gradient-free de los
    pesos del ansatz cuántico con COBYLA/SPSA (NO evolución diferencial).
  * Métricas de reconstrucción MSE/PSNR/SSIM.
  * Descriptores de expresibilidad/entrelazamiento del circuito reutilizando
    `../expressibility/qexpr.py` (Sim, Johnson & Aspuru-Guzik 2019).
  * `save_html_report`: informe HTML autocontenido por configuración (figuras embebidas
    en base64 + tablas), guardado en `results/<cfg.tag()>.html`.
  * `run_config`: orquesta un experimento completo (HQVAE + baseline) y devuelve una fila
    de métricas para el barrido.

Compatible con Qiskit 2.x + qiskit-machine-learning 0.9 (mismas factorías de circuito
que el resto del pipeline del TFM: zz_feature_map, pauli_feature_map, real_amplitudes,
efficient_su2).
"""

from __future__ import annotations

import os
import io
import sys
import math
import base64
import datetime
from dataclasses import dataclass, replace, asdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from skimage.metrics import structural_similarity as _ssim

from qiskit import QuantumCircuit
from qiskit.circuit.library import (
    zz_feature_map,
    pauli_feature_map,
    real_amplitudes,
    efficient_su2,
)
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


# --------------------------------------------------------------------------- #
# Reutilización de qexpr.py (métricas de expresibilidad/entrelazamiento, E6)
# --------------------------------------------------------------------------- #
from pathlib import Path
def _import_qexpr():
    """Importa qexpr buscando su raíz real desde este archivo hacia arriba."""
    try:
        import qexpr as qx
        return qx
    except ImportError:
        here = Path(__file__).resolve().parent

        candidates = [here, *here.parents]
        for base in candidates:
            direct = base / "qexpr.py"
            nested = base / "expressibility" / "qexpr.py"

            if direct.exists():
                sys.path.insert(0, str(base))
                import qexpr as qx
                return qx

            if nested.exists():
                sys.path.insert(0, str(base / "expressibility"))
                import qexpr as qx
                return qx

        raise ImportError(
            f"No se encontró qexpr.py desde {here} ni en sus ancestros."
        )
# ...existing code...


# --------------------------------------------------------------------------- #
# Ruido mixto (portado de REPOS/.../src/data/noise.py) — 3 clases
# --------------------------------------------------------------------------- #
def gaussian_noise(img: torch.Tensor, sigma: float = 0.25) -> torch.Tensor:
    return torch.clamp(img + torch.randn_like(img) * sigma, 0.0, 1.0)


def salt_pepper(img: torch.Tensor, prob: float = 0.15) -> torch.Tensor:
    noisy = img.clone()
    mask = torch.rand_like(img)
    noisy[mask < prob / 2] = 0.0
    noisy[mask > 1 - prob / 2] = 1.0
    return noisy


def speckle(img: torch.Tensor, sigma: float = 0.35) -> torch.Tensor:
    return torch.clamp(img + img * torch.randn_like(img) * sigma, 0.0, 1.0)


def apply_mixed_noise(img: torch.Tensor):
    """Aplica una de las 3 mezclas de ruido por pares. Devuelve (noisy, label).

      label 0 = gaussiano + speckle
      label 1 = gaussiano + sal y pimienta
      label 2 = sal y pimienta + speckle
    """
    r = int(torch.randint(0, 3, (1,)).item())
    if r == 0:
        return speckle(gaussian_noise(img)), 0
    elif r == 1:
        return salt_pepper(gaussian_noise(img)), 1
    else:
        return speckle(salt_pepper(img)), 2


class DenoisingMNISTDataset(Dataset):
    """Envuelve un dataset base de MNIST y devuelve (imagen_ruidosa, imagen_limpia).

    El ruido se precalcula en __init__ (con semilla) para que la evaluación sea
    reproducible entre épocas/runs.
    """

    def __init__(self, base_dataset, seed: int = 42):
        self.noisy = []
        self.clean = []
        g = torch.Generator().manual_seed(seed)
        # fija el estado global de torch para que el ruido sea determinista
        state = torch.get_rng_state()
        torch.manual_seed(seed)
        try:
            for i in range(len(base_dataset)):
                clean, _ = base_dataset[i]
                noisy, _ = apply_mixed_noise(clean)
                self.clean.append(clean)
                self.noisy.append(noisy)
        finally:
            torch.set_rng_state(state)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


def load_denoising_mnist(
    n_train: int = 300,
    n_val: int = 100,
    n_test: int = 100,
    batch_size: int = 16,
    seed: int = 42,
    data_root: Optional[str] = None,
):
    """Carga subconjuntos de MNIST y los convierte en datasets de denoising.

    Devuelve (train_loader, val_loader, test_loader, test_dataset).
    `data_root` por defecto apunta a ../data (MNIST ya descargado en TFM/data).
    """
    if data_root is None:
        data_root = os.path.join("..", "data")

    tfm = transforms.ToTensor()
    train_full = datasets.MNIST(root=data_root, train=True, download=True, transform=tfm)
    test_full = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)

    train_sub = Subset(train_full, list(range(n_train)))
    val_sub = Subset(train_full, list(range(n_train, n_train + n_val)))
    test_sub = Subset(test_full, list(range(n_test)))

    train_ds = DenoisingMNISTDataset(train_sub, seed=seed)
    val_ds = DenoisingMNISTDataset(val_sub, seed=seed + 1)
    test_ds = DenoisingMNISTDataset(test_sub, seed=seed + 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, test_ds


# --------------------------------------------------------------------------- #
# Configuración de un experimento
# --------------------------------------------------------------------------- #
_FM_TAG = {"zz": "ZZFM", "pauli": "PFM"}
_AN_TAG = {"real_amplitudes": "RA", "efficient_su2": "ESU2"}


@dataclass
class HQVAEConfig:
    # --- circuito cuántico ---
    n_qubits: int = 6                       # soportado 4 y 6 (y otros)
    feature_map: str = "zz"                 # "zz" | "pauli"
    ansatz: str = "real_amplitudes"         # "real_amplitudes" | "efficient_su2"
    fm_entanglement: str = "full"
    ansatz_entanglement: str = "linear"
    reps: int = 1
    paulis: Optional[list] = None           # para pauli_feature_map; def. ['X','Y','ZZ']
    # --- VAE ---
    latent_dim: int = 8                     # desacoplado de n_qubits (adaptador de ángulos)
    beta: float = 0.5
    # --- entrenamiento ---
    epochs: int = 15
    lr: float = 1e-3
    optimizer: str = "adam"                 # adam|adamw|sgd|rmsprop|lbfgs|cobyla|spsa
    batch_size: int = 16
    seed: int = 42

    def tag(self) -> str:
        fm = _FM_TAG.get(self.feature_map, self.feature_map.upper())
        an = _AN_TAG.get(self.ansatz, self.ansatz.upper())
        base = f"HQVAE{self.n_qubits}_{fm}_{an}_{self.ansatz_entanglement}_r{self.reps}"
        # sufijo del optimizador solo si no es el por defecto (evita colisiones al
        # comparar optimizadores sobre el mismo circuito; el barrido usa adam -> sin sufijo).
        if self.optimizer.lower() != "adam":
            base += f"_{self.optimizer.lower()}"
        return base


# --------------------------------------------------------------------------- #
# Circuito cuántico (EstimatorQNN)
# --------------------------------------------------------------------------- #
def _make_observables(n_qubits: int):
    """Z locales en cada qubit + términos ZZ en pares vecinos no solapados."""
    obs = []
    for k in range(n_qubits):
        label = ["I"] * n_qubits
        label[k] = "Z"
        obs.append(SparsePauliOp("".join(label)))
    for k in range(0, n_qubits - 1, 2):
        label = ["I"] * n_qubits
        label[k] = "Z"
        label[k + 1] = "Z"
        obs.append(SparsePauliOp("".join(label)))
    return obs


def _build_fm_ansatz(cfg: HQVAEConfig):
    if cfg.feature_map == "zz":
        fm = zz_feature_map(cfg.n_qubits, reps=1, entanglement=cfg.fm_entanglement)
    elif cfg.feature_map == "pauli":
        fm = pauli_feature_map(
            cfg.n_qubits,
            reps=1,
            paulis=cfg.paulis or ["X", "Y", "ZZ"],
            entanglement=cfg.fm_entanglement,
        )
    else:
        raise ValueError(f"feature_map desconocido: {cfg.feature_map}")

    if cfg.ansatz == "real_amplitudes":
        an = real_amplitudes(cfg.n_qubits, reps=cfg.reps, entanglement=cfg.ansatz_entanglement)
    elif cfg.ansatz == "efficient_su2":
        an = efficient_su2(cfg.n_qubits, reps=cfg.reps, entanglement=cfg.ansatz_entanglement)
    else:
        raise ValueError(f"ansatz desconocido: {cfg.ansatz}")
    return fm, an


def build_vae_qnn(cfg: HQVAEConfig):
    """Construye el EstimatorQNN del HQVAE. Devuelve (qnn, n_observables)."""
    fm, an = _build_fm_ansatz(cfg)
    qc = QuantumCircuit(cfg.n_qubits)
    qc.compose(fm, inplace=True)
    qc.compose(an, inplace=True)

    observables = _make_observables(cfg.n_qubits)
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=fm.parameters,
        weight_params=an.parameters,
        observables=observables,
        estimator=StatevectorEstimator(),
        input_gradients=True,
    )
    return qnn, len(observables)


# --------------------------------------------------------------------------- #
# Arquitecturas VAE
# --------------------------------------------------------------------------- #
def _make_encoder():
    # 1x28x28 -> 64*7*7 = 3136
    return nn.Sequential(
        nn.Conv2d(1, 32, 4, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(),
    )


def _make_decoder():
    # 3136 -> 1x28x28
    return nn.Sequential(
        nn.Unflatten(1, (64, 7, 7)),
        nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
        nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1), nn.Sigmoid(),
    )


class ConvVAE(nn.Module):
    """β-VAE convolucional clásico (baseline)."""

    def __init__(self, latent_dim: int = 8):
        super().__init__()
        self.encoder = _make_encoder()
        self.fc_mu = nn.Linear(3136, latent_dim)
        self.fc_logvar = nn.Linear(3136, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 3136)
        self.decoder = _make_decoder()

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        return self.decoder(self.fc_decode(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


class ConvHQVAE(nn.Module):
    """VAE convolucional híbrido: una capa cuántica (QNN) actúa sobre el latente.

    El latente `z` (dim latent_dim) se proyecta a `n_qubits` ángulos con un adaptador
    `fc_angles` + tanh()*pi (mismo encoding angular que el clasificador HQNN del TFM),
    de modo que latent_dim y n_qubits quedan desacoplados. La salida del QNN se concatena
    con `z` antes de decodificar (no se pierde el latente clásico).
    """

    def __init__(self, qnn, n_obs: int, latent_dim: int = 8, n_qubits: int = 6):
        super().__init__()
        self.n_qubits = n_qubits
        self.encoder = _make_encoder()
        self.fc_mu = nn.Linear(3136, latent_dim)
        self.fc_logvar = nn.Linear(3136, latent_dim)
        self.fc_angles = nn.Linear(latent_dim, n_qubits)   # adaptador latent_dim -> n_qubits
        self.qnn = TorchConnector(qnn)
        self.fc_decode = nn.Linear(latent_dim + n_obs, 3136)
        self.decoder = _make_decoder()

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z):
        angles = torch.tanh(self.fc_angles(z)) * math.pi
        z_qnn = self.qnn(angles)
        z_combined = torch.cat([z, z_qnn], dim=1)
        return self.decoder(self.fc_decode(z_combined))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta: float = 0.5):
    """Pérdida β-VAE = BCE(sum) + beta * KL."""
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + beta * kl


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
_TORCH_OPTIMIZERS = {"adam", "adamw", "sgd", "rmsprop", "lbfgs"}
_GRADIENT_FREE = {"cobyla", "spsa"}


def _make_optimizer(name: str, params, lr: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=1e-4, momentum=0.9,)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    if name == "lbfgs":
        return torch.optim.LBFGS(params, lr=lr)
    raise ValueError(f"Optimizador de PyTorch desconocido: {name}")


def _eval_loss(model, loader, beta, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            recon, mu, logvar = model(noisy)
            total += vae_loss(recon, clean, mu, logvar, beta).item()
            n += noisy.size(0)
    return total / max(n, 1)


def train_vae(model, train_loader, val_loader, cfg: HQVAEConfig, device="cpu", verbose=True):
    """Entrena un VAE (clásico o híbrido). Devuelve la historia de pérdidas.

    - Optimizadores de PyTorch (adam/adamw/sgd/rmsprop/lbfgs): entrenamiento end-to-end.
    - cobyla/spsa: entrena la parte clásica con Adam y luego refina SOLO los pesos del
      ansatz cuántico sin gradiente (experimental, lento).
    """
    model.to(device)
    opt_name = cfg.optimizer.lower()

    if opt_name in _GRADIENT_FREE:
        return _train_vae_gradient_free(model, train_loader, val_loader, cfg, device, verbose)

    optimizer = _make_optimizer(opt_name, model.parameters(), cfg.lr)
    history = {"train": [], "val": []}

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n = 0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            if opt_name == "lbfgs":
                def closure():
                    optimizer.zero_grad()
                    recon, mu, logvar = model(noisy)
                    loss = vae_loss(recon, clean, mu, logvar, cfg.beta)
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
            else:
                optimizer.zero_grad()
                recon, mu, logvar = model(noisy)
                loss = vae_loss(recon, clean, mu, logvar, cfg.beta)
                loss.backward()
                optimizer.step()

            running += float(loss.item())
            n += noisy.size(0)

        train_loss = running / max(n, 1)
        val_loss = _eval_loss(model, val_loader, cfg.beta, device)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if verbose:
            print(f"  epoch {epoch + 1:2d}/{cfg.epochs}  train={train_loss:.2f}  val={val_loss:.2f}")

    return history


def _get_qnn_weight(model):
    """Devuelve el Parameter de pesos del TorchConnector del HQVAE (o None)."""
    tc = getattr(model, "qnn", None)
    if tc is None:
        return None
    for attr in ("weight", "_weights"):
        w = getattr(tc, attr, None)
        if isinstance(w, torch.Tensor):
            return w
    return None


def _train_vae_gradient_free(model, train_loader, val_loader, cfg, device, verbose):
    """Fase 1: warm-up end-to-end con Adam. Fase 2: refina los pesos del ansatz
    cuántico con un optimizador sin gradiente (COBYLA/SPSA) minimizando la pérdida
    de validación. EXPERIMENTAL — pensado como estudio del efecto del optimizador."""
    from qiskit_machine_learning.optimizers import COBYLA, SPSA

    warm = replace(cfg, optimizer="adam")
    if verbose:
        print("  [gradient-free] fase 1: warm-up end-to-end con Adam")
    history = train_vae(model, train_loader, val_loader, warm, device, verbose)

    qnn_weight = _get_qnn_weight(model)
    if qnn_weight is None:
        if verbose:
            print("  [gradient-free] no se encontró el vector de pesos del QNN; se omite la fase 2.")
        return history

    if verbose:
        print(f"  [gradient-free] fase 2: refinado {cfg.optimizer.upper()} de {qnn_weight.numel()} pesos del ansatz")

    for p in model.parameters():
        p.requires_grad_(False)

    def objective(x):
        with torch.no_grad():
            qnn_weight.copy_(torch.tensor(np.asarray(x), dtype=qnn_weight.dtype, device=qnn_weight.device))
            return _eval_loss(model, val_loader, cfg.beta, device)

    x0 = qnn_weight.detach().cpu().numpy().ravel()
    optimizer = COBYLA(maxiter=cfg.epochs * 3) if cfg.optimizer.lower() == "cobyla" else SPSA(maxiter=cfg.epochs * 3)
    result = optimizer.minimize(fun=objective, x0=x0)

    with torch.no_grad():
        qnn_weight.copy_(torch.tensor(np.asarray(result.x), dtype=qnn_weight.dtype, device=qnn_weight.device))
    for p in model.parameters():
        p.requires_grad_(True)

    history["gradient_free"] = {"loss_final_val": float(result.fun)}
    if verbose:
        print(f"  [gradient-free] val loss tras refinado = {float(result.fun):.2f}")
    return history


# --------------------------------------------------------------------------- #
# Métricas de reconstrucción
# --------------------------------------------------------------------------- #
def eval_reconstruction(model, loader, device="cpu"):
    """MSE / PSNR / SSIM medios sobre el conjunto (imágenes normalizadas en [0,1])."""
    model.eval()
    mses, psnrs, ssims = [], [], []
    with torch.no_grad():
        for noisy, clean in loader:
            recon, _, _ = model(noisy.to(device))
            recon = recon.cpu().numpy().reshape(-1, 28, 28)
            clean = clean.numpy().reshape(-1, 28, 28)
            for i in range(clean.shape[0]):
                mse = float(((recon[i] - clean[i]) ** 2).mean())
                mses.append(mse)
                psnrs.append(10.0 * np.log10(1.0 / mse) if mse > 0 else 99.0)
                ssims.append(float(_ssim(clean[i], recon[i], data_range=1.0)))
    return {
        "mse": float(np.mean(mses)),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }


# --------------------------------------------------------------------------- #
# Descriptores de circuito (expresibilidad / entrelazamiento) — reutiliza qexpr
# --------------------------------------------------------------------------- #
def circuit_descriptors(cfg: HQVAEConfig, n_samples: int = 1500, seed: int = 42) -> dict:
    """KL de expresibilidad y Q de Meyer-Wallach del circuito de la config (E6)."""
    qx = _import_qexpr()
    qc, _ = qx.build_pqc(
        feature_map=cfg.feature_map,
        ansatz=cfg.ansatz,
        n_qubits=cfg.n_qubits,
        reps=cfg.reps,
        fm_entanglement=cfg.fm_entanglement,
        ansatz_entanglement=cfg.ansatz_entanglement,
        paulis=cfg.paulis,
    )
    return qx.descriptors(qc, cfg.n_qubits, n_samples=n_samples, seed=seed)


# --------------------------------------------------------------------------- #
# Figuras (devuelven un objeto Figure; no llaman a plt.show)
# --------------------------------------------------------------------------- #
def fig_reconstruction_grid(hqvae, vae, dataset, n=8, device="cpu", seed=0):
    """Filas: Limpia / Ruidosa / HQVAE / ConvVAE (si se pasa)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    n = len(idx)
    noisy = torch.stack([dataset[i][0] for i in idx])
    clean = torch.stack([dataset[i][1] for i in idx])

    rows = [("Limpia", clean), ("Ruidosa", noisy)]
    hqvae.eval()
    with torch.no_grad():
        recon_h, _, _ = hqvae(noisy.to(device))
        rows.append(("HQVAE", recon_h.cpu()))
        if vae is not None:
            vae.eval()
            recon_v, _, _ = vae(noisy.to(device))
            rows.append(("ConvVAE", recon_v.cpu()))

    nrows = len(rows)
    fig, axes = plt.subplots(nrows, n, figsize=(1.4 * n, 1.5 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, 0)
    if n == 1:
        axes = np.expand_dims(axes, 1)
    for r, (label, imgs) in enumerate(rows):
        imgs = imgs.detach().cpu().numpy().reshape(-1, 28, 28)
        for c in range(n):
            ax = axes[r, c]
            ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(label, fontsize=11, rotation=90)
    fig.tight_layout()
    return fig


def fig_loss_curves(hist_h, hist_v=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist_h["train"], label="HQVAE train")
    ax.plot(hist_h["val"], label="HQVAE val")
    if hist_v is not None:
        ax.plot(hist_v["train"], "--", label="ConvVAE train")
        ax.plot(hist_v["val"], "--", label="ConvVAE val")
    ax.set_xlabel("época")
    ax.set_ylabel("pérdida (BCE + βKL) por muestra")
    ax.set_title("Curvas de entrenamiento")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_circuit(cfg: HQVAEConfig):
    """Dibuja el circuito (feature map + ansatz). Requiere pylatexenc."""
    qx = _import_qexpr()
    qc, _ = qx.build_pqc(
        feature_map=cfg.feature_map,
        ansatz=cfg.ansatz,
        n_qubits=cfg.n_qubits,
        reps=cfg.reps,
        fm_entanglement=cfg.fm_entanglement,
        ansatz_entanglement=cfg.ansatz_entanglement,
        paulis=cfg.paulis,
    )
    return qc.decompose().draw("mpl", fold=-1)


# --------------------------------------------------------------------------- #
# Informe HTML autocontenido
# --------------------------------------------------------------------------- #
def _fig_to_base64(fig, dpi=110) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


_HTML_CSS = """
<style>
  body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 32px; color: #1a1a1a; max-width: 1100px; }
  h1 { border-bottom: 2px solid #444; padding-bottom: 6px; }
  h2 { margin-top: 28px; color: #333; }
  table { border-collapse: collapse; margin: 8px 0 18px 0; }
  th, td { border: 1px solid #ccc; padding: 5px 12px; text-align: right; }
  th { background: #f0f0f0; }
  td:first-child, th:first-child { text-align: left; }
  img { max-width: 100%; height: auto; border: 1px solid #eee; margin: 6px 0; }
  .meta { color: #666; font-size: 0.85em; }
</style>
"""


def save_html_report(cfg, metrics_hqvae, metrics_vae, descriptors, figs, out_dir="results"):
    """Escribe results/<cfg.tag()>.html autocontenido (figuras en base64 + tablas)."""
    os.makedirs(out_dir, exist_ok=True)

    # tabla de configuración
    cfg_df = pd.DataFrame(list(asdict(cfg).items()), columns=["parámetro", "valor"])

    # tabla de métricas clásico vs híbrido
    cols = {"HQVAE (híbrido)": metrics_hqvae}
    if metrics_vae is not None:
        cols["ConvVAE (clásico)"] = metrics_vae
    metrics_df = pd.DataFrame(
        {name: [m["mse"], m["psnr"], m["ssim"]] for name, m in cols.items()},
        index=["MSE", "PSNR (dB)", "SSIM"],
    )

    # tabla de descriptores del circuito
    desc_df = pd.DataFrame(list(descriptors.items()), columns=["descriptor", "valor"])

    imgs_html = ""
    for title, fig in figs.items():
        if fig is None:
            continue
        imgs_html += f"<h2>{title}</h2>\n<img src='data:image/png;base64,{_fig_to_base64(fig)}'/>\n"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"><title>{cfg.tag()}</title>{_HTML_CSS}</head>
<body>
<h1>HQVAE denoising — {cfg.tag()}</h1>
<p class="meta">MNIST con ruido mixto · reconstrucción · generado {now}</p>

<h2>Métricas de reconstrucción (test)</h2>
{metrics_df.to_html(float_format=lambda v: f"{v:.4f}")}

<h2>Descriptores del circuito (Sim et al. 2019)</h2>
{desc_df.to_html(index=False, float_format=lambda v: f"{v:.4f}")}

{imgs_html}

<h2>Configuración</h2>
{cfg_df.to_html(index=False)}
</body></html>
"""
    path = os.path.join(out_dir, cfg.tag() + ".html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


# --------------------------------------------------------------------------- #
# Orquestación de un experimento completo
# --------------------------------------------------------------------------- #
def run_config(
    cfg: HQVAEConfig,
    loaders,
    device="cpu",
    results_dir="results",
    train_baseline=True,
    n_grid_show=8,
    draw_circuit=True,
    verbose=True,
):
    """Entrena HQVAE (+ ConvVAE baseline), evalúa, genera figuras y guarda el HTML.

    Devuelve un dict con la fila de métricas ('row'), la ruta del HTML ('html') y los modelos.
    """
    train_loader, val_loader, test_loader, test_ds = loaders
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if verbose:
        print(f"[{cfg.tag()}] entrenando HQVAE ({cfg.optimizer})...")
    qnn, n_obs = build_vae_qnn(cfg)
    hqvae = ConvHQVAE(qnn, n_obs, latent_dim=cfg.latent_dim, n_qubits=cfg.n_qubits)
    hist_h = train_vae(hqvae, train_loader, val_loader, cfg, device, verbose)
    m_h = eval_reconstruction(hqvae, test_loader, device)

    vae, hist_v, m_v = None, None, None
    if train_baseline:
        if verbose:
            print(f"[{cfg.tag()}] entrenando ConvVAE baseline (adam)...")
        vae = ConvVAE(latent_dim=cfg.latent_dim)
        hist_v = train_vae(vae, train_loader, val_loader, replace(cfg, optimizer="adam"), device, verbose)
        m_v = eval_reconstruction(vae, test_loader, device)

    if verbose:
        print(f"[{cfg.tag()}] calculando descriptores de expresibilidad/entrelazamiento...")
    desc = circuit_descriptors(cfg)

    figs = {
        "Reconstrucciones (Limpia / Ruidosa / HQVAE / ConvVAE)":
            fig_reconstruction_grid(hqvae, vae, test_ds, n=n_grid_show, device=device),
        "Curvas de entrenamiento": fig_loss_curves(hist_h, hist_v),
    }
    if draw_circuit:
        try:
            figs["Circuito (feature map + ansatz)"] = fig_circuit(cfg)
        except Exception as exc:  # pylatexenc ausente u otro problema de dibujo
            if verbose:
                print(f"  (no se pudo dibujar el circuito: {exc})")

    html_path = save_html_report(cfg, m_h, m_v, desc, figs, results_dir)
    if verbose:
        print(f"[{cfg.tag()}] HTML -> {html_path}")

    row = {
        "tag": cfg.tag(),
        "n_qubits": cfg.n_qubits,
        "feature_map": cfg.feature_map,
        "ansatz": cfg.ansatz,
        "entanglement": cfg.ansatz_entanglement,
        "reps": cfg.reps,
        "optimizer": cfg.optimizer,
        "kl": desc.get("expressibility_kl"),
        "Q": desc.get("entangling_Q"),
        "mse_hqvae": m_h["mse"], "psnr_hqvae": m_h["psnr"], "ssim_hqvae": m_h["ssim"],
    }
    if m_v is not None:
        row.update({"mse_vae": m_v["mse"], "psnr_vae": m_v["psnr"], "ssim_vae": m_v["ssim"]})

    return {"row": row, "html": html_path, "hqvae": hqvae, "vae": vae,
            "hist_hqvae": hist_h, "hist_vae": hist_v}


# --------------------------------------------------------------------------- #
# Barrido por defecto (guiado por E6 / Sim et al. 2019)
# --------------------------------------------------------------------------- #
def default_sweep(latent_dim=8, beta=0.5, epochs=15, seed=42):
    """Grid definido en el plan: 4 y 6 qubits × feature map × ansatz × entanglement × reps."""
    base = dict(latent_dim=latent_dim, beta=beta, epochs=epochs, seed=seed)
    return [
        HQVAEConfig(n_qubits=6, feature_map="zz", ansatz="real_amplitudes",
                    ansatz_entanglement="linear", reps=1, **base),
        HQVAEConfig(n_qubits=4, feature_map="zz", ansatz="real_amplitudes",
                    ansatz_entanglement="linear", reps=1, **base),
        HQVAEConfig(n_qubits=6, feature_map="zz", ansatz="real_amplitudes",
                    ansatz_entanglement="full", reps=1, **base),
        HQVAEConfig(n_qubits=6, feature_map="zz", ansatz="efficient_su2",
                    ansatz_entanglement="linear", reps=1, **base),
        HQVAEConfig(n_qubits=4, feature_map="pauli", ansatz="real_amplitudes",
                    ansatz_entanglement="linear", reps=1, **base),
        HQVAEConfig(n_qubits=6, feature_map="zz", ansatz="real_amplitudes",
                    ansatz_entanglement="linear", reps=2, **base),
    ]
