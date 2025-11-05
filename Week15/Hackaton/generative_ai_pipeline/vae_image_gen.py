"""Variational autoencoder utilities for optional image generation."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Set tokenizers parallelism before any imports to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from config import IMAGE_OUTPUT_DIR, MODEL_DIR
from logging_utils import configure_logging

logger = configure_logging(__name__)

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    HAS_TORCH = True
except Exception:  # pragma: no cover - optional dependency
    HAS_TORCH = False
    torch = nn = optim = DataLoader = Subset = datasets = transforms = save_image = None  # type: ignore


if HAS_TORCH:

    class Encoder(nn.Module):
        """Simple convolutional encoder."""

        def __init__(self, latent_dim: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
            )
            self.fc_mu = nn.Linear(64 * 8 * 8, latent_dim)
            self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

        def forward(self, x):  # type: ignore[override]
            h = self.conv(x)
            return self.fc_mu(h), self.fc_logvar(h)


    class Decoder(nn.Module):
        """Simple convolutional decoder."""

        def __init__(self, latent_dim: int) -> None:
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(latent_dim, 64 * 8 * 8),
                nn.ReLU(inplace=True),
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid(),
            )

        def forward(self, z):  # type: ignore[override]
            h = self.fc(z)
            h = h.view(-1, 64, 8, 8)
            return self.deconv(h)


    class VAE(nn.Module):
        """Variational autoencoder model for 32x32 images."""

        def __init__(self, latent_dim: int) -> None:
            super().__init__()
            self.encoder = Encoder(latent_dim)
            self.decoder = Decoder(latent_dim)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):  # type: ignore[override]
            mu, logvar = self.encoder(x)
            z = self.reparameterize(mu, logvar)
            recon = self.decoder(z)
            return recon, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):  # pragma: no cover - heavy computation
    """Compute standard VAE loss (reconstruction + KL divergence)."""
    bce = nn.functional.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / x.size(0)


@dataclass
class VAEGenerator:
    """Utility class encapsulating training and sampling for a lightweight VAE."""

    latent_dim: int = 128
    checkpoint_name: str = "vae_cifar10.pth"

    def __post_init__(self) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "torch and torchvision are required for VAE functionality. "
                "Install them via `pip install torch torchvision`."
            )
        # Force CPU usage to avoid CUDA compatibility issues
        self.device = torch.device("cpu")
        self.model = VAE(self.latent_dim).to(self.device)
        self.checkpoint_path = MODEL_DIR / self.checkpoint_name
        logger.info(
            "Initialized VAEGenerator",
            extra={"latent_dim": self.latent_dim, "device": str(self.device), "checkpoint": str(self.checkpoint_path)},
        )

    def load_checkpoint(self) -> bool:
        """Load model weights if checkpoint is available."""
        if self.checkpoint_path.exists():
            state_dict = torch.load(self.checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            logger.info("Loaded VAE checkpoint", extra={"path": str(self.checkpoint_path)})
            return True
        logger.warning("VAE checkpoint not found", extra={"expected_path": str(self.checkpoint_path)})
        return False

    def train_vae(
        self,
        epochs: int = 20,
        batch_size: int = 128,
        sample_pct: float = 0.05,
        learning_rate: float = 1e-3,
    ) -> Path:
        """Train the VAE on a CIFAR-10 subset and persist the checkpoint."""
        transform = transforms.Compose([transforms.ToTensor()])
        dataset = datasets.CIFAR10(root=str(MODEL_DIR), train=True, download=True, transform=transform)

        sample_size = max(1, int(len(dataset) * sample_pct))
        indices = list(range(len(dataset)))[:sample_size]
        subset = Subset(dataset, indices)

        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(epochs):  # pragma: no cover - heavy computation
            total_loss = 0.0
            for batch in dataloader:
                images, _ = batch
                images = images.to(self.device)
                optimizer.zero_grad()
                recon, mu, logvar = self.model(images)
                loss = vae_loss_function(recon, images, mu, logvar)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(dataloader)
            logger.info("VAE epoch complete", extra={"epoch": epoch + 1, "avg_loss": avg_loss})

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.checkpoint_path)
        logger.info("Saved VAE checkpoint", extra={"path": str(self.checkpoint_path)})
        return self.checkpoint_path

    def decode(self, latent_vector) -> torch.Tensor:
        """Decode a latent vector into an image tensor."""
        self.model.eval()
        with torch.no_grad():
            latent_vector = latent_vector.to(self.device)
            image = self.model.decoder(latent_vector)
        return image

    def generate(self, num_samples: int = 8) -> list[Path]:
        """Sample random latent vectors and persist generated images."""
        self.model.eval()
        IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        with torch.no_grad():
            for idx in range(num_samples):
                latent = torch.randn(1, self.latent_dim, device=self.device)
                image = self.model.decoder(latent).cpu()
                filename = f"generated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{idx}.png"
                path = IMAGE_OUTPUT_DIR / filename
                save_image(image, str(path))
                saved_paths.append(path)
                logger.debug("Generated VAE image", extra={"path": str(path)})

        return saved_paths
