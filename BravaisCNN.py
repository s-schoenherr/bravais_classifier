"""Train a small PyTorch CNN to classify synthetic LEED images.

The script generates data on the fly with LEEDGenerator, trains a compact CNN,
and evaluates it on an independent synthetic test set.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except ModuleNotFoundError as exc:  # pragma: no cover - friendly runtime error
    raise SystemExit(
        "PyTorch is not installed. Install the project dependencies first, for example:\n"
        "  uv sync\n"
        "Then run:\n"
        "  uv run python BravaisCNN.py"
    ) from exc

from LEEDGenerator import LEEDGenerator


@dataclass(frozen=True)
class TrainConfig:
    img_size: int = 128
    train_samples: int = 3000
    test_samples: int = 750
    batch_size: int = 64
    epochs: int = 8
    lr: float = 1e-3
    method: str = "reciprocal"
    seed: int = 7
    device: str = "auto"
    save_model: Path | None = Path("bravais_cnn.pt")


class SyntheticLEEDDataset(Dataset):
    """Balanced synthetic dataset generated once and kept in memory."""

    def __init__(self, n_samples: int, img_size: int, method: str, seed: int):
        self.n_samples = n_samples
        self.img_size = img_size
        self.method = method
        self.seed = seed
        self.classes = LEEDGenerator.classes()
        self.images = np.empty((n_samples, 1, img_size, img_size), dtype=np.float32)
        self.labels = np.empty((n_samples,), dtype=np.int64)

        for index in range(n_samples):
            image, label = self._generate_sample(index)
            self.images[index, 0] = image
            self.labels[index] = label

    def __len__(self) -> int:
        return self.n_samples

    def _generate_sample(self, index: int) -> tuple[np.ndarray, int]:
        rng_state = np.random.get_state()
        np.random.seed(self.seed + index)
        try:
            label = index % len(self.classes)
            bravais = self.classes[label]
            method = self.method
            if method == "mixed":
                method = "fft" if np.random.random() < 0.5 else "reciprocal"

            gen = LEEDGenerator(img_size=self.img_size, random_rotation=True)
            image = gen.generate(bravais, method=method).astype(np.float32)
        finally:
            np.random.set_state(rng_state)

        return image, label

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[index])
        target = torch.tensor(int(self.labels[index]), dtype=torch.long)
        return image, target


class BravaisCNN(nn.Module):
    """Small CNN designed for speed on a laptop CPU."""

    def __init__(self, n_classes: int = 5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.GroupNorm(12, 96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    n_classes: int,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)
        predictions = logits.argmax(dim=1)

        running_loss += float(loss.item()) * labels.size(0)
        correct += int((predictions == labels).sum().item())
        total += labels.size(0)

        for truth, pred in zip(labels.cpu().numpy(), predictions.cpu().numpy()):
            confusion[int(truth), int(pred)] += 1

    return running_loss / total, correct / total, confusion


def print_confusion_matrix(confusion: np.ndarray, classes: Iterable[str]) -> None:
    labels = list(classes)
    width = max(8, max(len(label) for label in labels) + 1)
    print("\nConfusion matrix: rows=true, columns=predicted")
    print(" " * width + "".join(label[:7].rjust(8) for label in labels))
    for label, row in zip(labels, confusion):
        print(label[: width - 1].ljust(width) + "".join(str(int(v)).rjust(8) for v in row))

    print("\nPer-class accuracy:")
    for label, row in zip(labels, confusion):
        total = row.sum()
        accuracy = row[labels.index(label)] / total if total else 0.0
        print(f"  {label:22s} {accuracy:6.2%} ({row[labels.index(label)]}/{total})")


def train(config: TrainConfig) -> tuple[BravaisCNN, np.ndarray, float]:
    set_seed(config.seed)
    device = resolve_device(config.device)
    classes = LEEDGenerator.classes()

    print("Generating synthetic train/test data...")
    train_data = SyntheticLEEDDataset(
        n_samples=config.train_samples,
        img_size=config.img_size,
        method=config.method,
        seed=config.seed,
    )
    test_data = SyntheticLEEDDataset(
        n_samples=config.test_samples,
        img_size=config.img_size,
        method=config.method,
        seed=config.seed + 1_000_000,
    )

    train_loader = make_loader(train_data, config.batch_size, shuffle=True)
    test_loader = make_loader(test_data, config.batch_size, shuffle=False)

    model = BravaisCNN(n_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on {device} with {len(train_data)} train and {len(test_data)} test samples.")
    print(f"Classes: {', '.join(classes)}")

    best_accuracy = 0.0
    best_confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for epoch in range(1, config.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_accuracy, confusion = evaluate(
            model, test_loader, loss_fn, device, n_classes=len(classes)
        )

        if test_accuracy >= best_accuracy:
            best_accuracy = test_accuracy
            best_confusion = confusion

        print(
            f"epoch {epoch:02d}/{config.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.2%} "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.2%}"
        )

    print_confusion_matrix(best_confusion, classes)

    if config.save_model is not None:
        torch.save(
            {
                "model_state": model.state_dict(),
                "classes": classes,
                "config": vars(config),
                "best_test_accuracy": best_accuracy,
                "confusion_matrix": best_confusion,
            },
            config.save_model,
        )
        print(f"\nSaved model checkpoint to {config.save_model}")

    return model, best_confusion, best_accuracy


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--train-samples", type=int, default=3000)
    parser.add_argument("--test-samples", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--method",
        choices=("reciprocal", "fft", "mixed"),
        default="reciprocal",
        help="Synthetic data model. 'mixed' trains on both reciprocal spots and FFT images.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--save-model", type=Path, default=Path("bravais_cnn.pt"))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        img_size=args.img_size,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        method=args.method,
        seed=args.seed,
        device=args.device,
        save_model=None if args.no_save else args.save_model,
    )


if __name__ == "__main__":
    train(parse_args())
