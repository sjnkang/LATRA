import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split


# ============================================================
# 1. Reproducibility
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 완전한 재현성을 위해 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 2. ResNet-8
# ============================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet8(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()

        # Stem
        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # Residual stages
        self.layer1 = BasicBlock(
            16,
            16,
            stride=1
        )

        self.layer2 = BasicBlock(
            16,
            32,
            stride=2
        )

        self.layer3 = BasicBlock(
            32,
            64,
            stride=2
        )

        # Classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.fc(x)

        return x


# ============================================================
# 3. CIFAR-10 Dataset
# ============================================================

def get_cifar10_loaders(
    batch_size=128,
    val_ratio=0.1,
    num_workers=2
):

    # --------------------------------------------------------
    # Training augmentation
    # --------------------------------------------------------

    train_transform = transforms.Compose([
        transforms.RandomCrop(
            32,
            padding=4
        ),

        transforms.RandomHorizontalFlip(),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # --------------------------------------------------------
    # Validation / Test
    # --------------------------------------------------------

    eval_transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010)
        )
    ])

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    full_trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    # Validation을 위해 분리
    val_size = int(
        len(full_trainset) * val_ratio
    )

    train_size = len(full_trainset) - val_size

    trainset, valset = random_split(
        full_trainset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Validation에서는 augmentation을 사용하지 않도록 설정
    valset.dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=eval_transform
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=eval_transform
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    testloader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return trainloader, valloader, testloader


# ============================================================
# 4. Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    epoch_loss = running_loss / total

    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


# ============================================================
# 5. Validation / Test
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = outputs.max(1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    loss = running_loss / total

    accuracy = 100.0 * correct / total

    return loss, accuracy


# ============================================================
# 6. Model Statistics
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def get_model_size_mb(model):

    param_size = 0

    buffer_size = 0

    for param in model.parameters():
        param_size += (
            param.nelement()
            * param.element_size()
        )

    for buffer in model.buffers():
        buffer_size += (
            buffer.nelement()
            * buffer.element_size()
        )

    size_mb = (
        param_size + buffer_size
    ) / (1024 ** 2)

    return size_mb


# ============================================================
# 7. Save Checkpoint
# ============================================================

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    val_accuracy,
    path
):

    checkpoint = {
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scheduler_state_dict":
            scheduler.state_dict()
            if scheduler is not None
            else None,

        "val_accuracy":
            val_accuracy
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# 8. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    SEED = 42

    BATCH_SIZE = 128

    EPOCHS = 150

    LEARNING_RATE = 0.1

    WEIGHT_DECAY = 5e-4

    VAL_RATIO = 0.1

    CHECKPOINT_DIR = "./checkpoints"

    CHECKPOINT_PATH = os.path.join(
        CHECKPOINT_DIR,
        "resnet8_cifar10_best.pth"
    )

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)

    print("LATRA - ResNet-8 CIFAR-10 Training")

    print("=" * 60)

    print(f"Device: {device}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading CIFAR-10...")

    trainloader, valloader, testloader = (
        get_cifar10_loaders(
            batch_size=BATCH_SIZE,
            val_ratio=VAL_RATIO
        )
    )

    print(
        f"Train samples: "
        f"{len(trainloader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(valloader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(testloader.dataset)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ResNet8(
        num_classes=10
    ).to(device)

    # --------------------------------------------------------
    # Baseline statistics
    # --------------------------------------------------------

    parameters = count_parameters(
        model
    )

    model_size = get_model_size_mb(
        model
    )

    print("\nModel Information")

    print("-" * 60)

    print(
        f"Trainable Parameters: "
        f"{parameters:,}"
    )

    print(
        f"FP32 Model Size: "
        f"{model_size:.2f} MB"
    )

    # --------------------------------------------------------
    # Loss / Optimizer
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Learning Rate Scheduler
    # --------------------------------------------------------

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_accuracy = 0.0

    best_epoch = 0

    print("\nStarting Training...")

    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model,
            trainloader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = evaluate(
            model,
            valloader,
            criterion,
            device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch "
            f"[{epoch:03d}/{EPOCHS}] | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc

            best_epoch = epoch

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_acc,
                CHECKPOINT_PATH
            )

            print(
                f"  -> Best checkpoint saved "
                f"(Val Acc: {val_acc:.2f}%)"
            )

        scheduler.step()

    # ========================================================
    # Load Best Checkpoint
    # ========================================================

    print("\n" + "=" * 60)

    print("Loading Best Checkpoint")

    print("=" * 60)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Best Epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best Validation Accuracy: "
        f"{checkpoint['val_accuracy']:.2f}%"
    )

    # ========================================================
    # Final Test Evaluation
    # ========================================================

    print("\n" + "=" * 60)

    print("Final Test Evaluation")

    print("=" * 60)

    test_loss, test_accuracy = evaluate(
        model,
        testloader,
        criterion,
        device
    )

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    # ========================================================
    # Final Model Information
    # ========================================================

    print("\n" + "=" * 60)

    print("Original ResNet-8 Baseline")

    print("=" * 60)

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    print(
        f"Model Size: "
        f"{get_model_size_mb(model):.2f} MB"
    )

    print(
        f"Validation Accuracy: "
        f"{best_val_accuracy:.2f}%"
    )

    print(
        f"Test Accuracy: "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Best Epoch: "
        f"{best_epoch}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_PATH}"
    )

    print("\nTraining completed.")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
