import os
import csv
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# ============================================================
# Configuration
# ============================================================

CHECKPOINT_PATH = "./checkpoints/resnet8_cifar10_best.pth"
RANK_CSV_PATH = "./analysis/latra_selected_ranks.csv"

OUTPUT_CSV_PATH = "./analysis/latra_full_model_result.csv"

BATCH_SIZE = 128
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# ResNet-8
# ============================================================

class BasicBlock(nn.Module):
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

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = torch.relu(out)

        return out


class ResNet8(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = BasicBlock(16, 16, stride=1)
        self.layer2 = BasicBlock(16, 32, stride=2)
        self.layer3 = BasicBlock(32, 64, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


# ============================================================
# Load Checkpoint
# ============================================================

def load_checkpoint(model, path):

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)

    return model


# ============================================================
# CIFAR-10
# ============================================================

def get_test_loader():

    transform = transforms.Compose([
        transforms.ToTensor(),

        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return loader


# ============================================================
# Tucker Decomposition
# ============================================================

def tucker_reconstruct(weight, r_out, r_in):

    """
    Tucker decomposition for Conv2D weight.

    Original:
        [out_channels, in_channels, kernel_h, kernel_w]

    Tucker rank:
        [r_out, r_in, kernel_h, kernel_w]
    """

    device = weight.device

    O, I, KH, KW = weight.shape

    # --------------------------------------------------------
    # Mode-0 unfolding
    # --------------------------------------------------------

    W = weight.permute(0, 1, 2, 3).contiguous()

    W0 = W.reshape(O, -1)

    U0, S0, V0 = torch.linalg.svd(
        W0,
        full_matrices=False
    )

    r_out = min(r_out, U0.shape[1])

    U0 = U0[:, :r_out]

    # --------------------------------------------------------
    # Mode-1 unfolding
    # --------------------------------------------------------

    W1 = W.permute(1, 0, 2, 3).contiguous()
    W1 = W1.reshape(I, -1)

    U1, S1, V1 = torch.linalg.svd(
        W1,
        full_matrices=False
    )

    r_in = min(r_in, U1.shape[1])

    U1 = U1[:, :r_in]

    # --------------------------------------------------------
    # Core tensor
    # --------------------------------------------------------

    core = torch.einsum(
        "oa,ib,oijk->abjk",
        U0,
        U1,
        W
    )

    # --------------------------------------------------------
    # Reconstruct
    # --------------------------------------------------------

    reconstructed = torch.einsum(
        "oa,ib,abjk->oijk",
        U0,
        U1,
        core
    )

    return reconstructed.to(device)


# ============================================================
# Load LATRA ranks
# ============================================================

def load_latra_ranks(path):

    ranks = {}

    with open(path, "r", newline="") as f:

        reader = csv.DictReader(f)

        for row in reader:

            layer = row["Layer"]

            r_out = int(row["r_out"])
            r_in = int(row["r_in"])

            ranks[layer] = (r_out, r_in)

    return ranks


# ============================================================
# Get convolution module
# ============================================================

def get_layer(model, layer_name):

    modules = {
        "conv1": model.conv1,

        "layer1.conv1": model.layer1.conv1,
        "layer1.conv2": model.layer1.conv2,

        "layer2.conv1": model.layer2.conv1,
        "layer2.conv2": model.layer2.conv2,

        "layer3.conv1": model.layer3.conv1,
        "layer3.conv2": model.layer3.conv2,
    }

    return modules[layer_name]


# ============================================================
# Apply LATRA
# ============================================================

def apply_latra(model, ranks):

    reconstruction_errors = {}

    for layer_name, (r_out, r_in) in ranks.items():

        layer = get_layer(
            model,
            layer_name
        )

        original_weight = layer.weight.data.clone()

        reconstructed_weight = tucker_reconstruct(
            original_weight,
            r_out,
            r_in
        )

        # ----------------------------------------------------
        # Reconstruction error
        # ----------------------------------------------------

        error = torch.norm(
            original_weight - reconstructed_weight
        ) / torch.norm(original_weight)

        reconstruction_errors[layer_name] = error.item()

        # ----------------------------------------------------
        # Replace weight
        # ----------------------------------------------------

        layer.weight.data.copy_(
            reconstructed_weight
        )

        print(
            f"{layer_name:18s} "
            f"rank=({r_out},{r_in}) "
            f"error={error.item():.6f}"
        )

    return model, reconstruction_errors


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, loader):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += (
            loss.item() * labels.size(0)
        )

        _, predicted = outputs.max(1)

        correct += (
            predicted == labels
        ).sum().item()

        total += labels.size(0)

    loss = total_loss / total
    accuracy = 100.0 * correct / total

    return loss, accuracy


# ============================================================
# Parameter statistics
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
    )


def count_conv_parameters(model):

    return sum(
        p.numel()
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
        for p in module.parameters()
    )


def calculate_tucker_parameter_count(
    model,
    ranks
):

    total = 0

    for layer_name, (r_out, r_in) in ranks.items():

        layer = get_layer(
            model,
            layer_name
        )

        O, I, KH, KW = layer.weight.shape

        # Tucker:
        #
        # U_out : O * r_out
        # U_in  : I * r_in
        # Core  : r_out * r_in * KH * KW

        params = (
            O * r_out
            + I * r_in
            + r_out * r_in * KH * KW
        )

        total += params

    return total


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 80)
    print("LATRA - Full Model Evaluation")
    print("=" * 80)

    print()
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Loading CIFAR-10 Test Set")
    print("=" * 80)

    test_loader = get_test_loader()

    print(
        f"Test samples: "
        f"{len(test_loader.dataset):,}"
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Baseline Model")
    print("=" * 80)

    baseline_model = ResNet8(
        num_classes=10
    ).to(DEVICE)

    baseline_model = load_checkpoint(
        baseline_model,
        CHECKPOINT_PATH
    )

    baseline_loss, baseline_accuracy = evaluate(
        baseline_model,
        test_loader
    )

    print(
        f"Baseline Loss: "
        f"{baseline_loss:.4f}"
    )

    print(
        f"Baseline Accuracy: "
        f"{baseline_accuracy:.2f}%"
    )

    baseline_params = count_parameters(
        baseline_model
    )

    print(
        f"Baseline Parameters: "
        f"{baseline_params:,}"
    )

    # --------------------------------------------------------
    # Load ranks
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Loading LATRA Ranks")
    print("=" * 80)

    ranks = load_latra_ranks(
        RANK_CSV_PATH
    )

    for layer_name, rank in ranks.items():

        print(
            f"{layer_name:18s} "
            f"rank={rank}"
        )

    # --------------------------------------------------------
    # Fresh model
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Applying LATRA")
    print("=" * 80)

    latra_model = ResNet8(
        num_classes=10
    ).to(DEVICE)

    latra_model = load_checkpoint(
        latra_model,
        CHECKPOINT_PATH
    )

    latra_model, reconstruction_errors = apply_latra(
        latra_model,
        ranks
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("LATRA Full Model Evaluation")
    print("=" * 80)

    latra_loss, latra_accuracy = evaluate(
        latra_model,
        test_loader
    )

    accuracy_drop = (
        baseline_accuracy
        - latra_accuracy
    )

    print()
    print(
        f"Baseline Accuracy : "
        f"{baseline_accuracy:.2f}%"
    )

    print(
        f"LATRA Accuracy    : "
        f"{latra_accuracy:.2f}%"
    )

    print(
        f"Accuracy Drop     : "
        f"{accuracy_drop:.2f}%"
    )

    print(
        f"Baseline Loss     : "
        f"{baseline_loss:.4f}"
    )

    print(
        f"LATRA Loss        : "
        f"{latra_loss:.4f}"
    )

    # --------------------------------------------------------
    # Parameter analysis
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("Parameter Analysis")
    print("=" * 80)

    original_conv_params = 0
    tucker_params = 0

    for layer_name, (r_out, r_in) in ranks.items():

        layer = get_layer(
            baseline_model,
            layer_name
        )

        O, I, KH, KW = layer.weight.shape

        original = O * I * KH * KW

        compressed = (
            O * r_out
            + I * r_in
            + r_out * r_in * KH * KW
        )

        original_conv_params += original
        tucker_params += compressed

        reduction = (
            1.0
            - compressed / original
        ) * 100.0

        print(
            f"{layer_name:18s} "
            f"original={original:,} "
            f"tucker={compressed:,} "
            f"reduction={reduction:.2f}%"
        )

    # --------------------------------------------------------
    # Full model parameter reduction
    #
    # Only convolution weights are actually replaced
    # by Tucker representation conceptually.
    # Other parameters remain unchanged.
    # --------------------------------------------------------

    total_original_params = baseline_params

    total_new_params = (
        total_original_params
        - original_conv_params
        + tucker_params
    )

    total_reduction = (
        1.0
        - total_new_params / total_original_params
    ) * 100.0

    print()
    print(
        f"Original Model Parameters : "
        f"{total_original_params:,}"
    )

    print(
        f"LATRA Estimated Parameters: "
        f"{total_new_params:,}"
    )

    print(
        f"Overall Parameter Reduction: "
        f"{total_reduction:.2f}%"
    )

    # --------------------------------------------------------
    # Average reconstruction error
    # --------------------------------------------------------

    avg_error = sum(
        reconstruction_errors.values()
    ) / len(reconstruction_errors)

    print()
    print(
        f"Average Reconstruction Error: "
        f"{avg_error:.6f}"
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_CSV_PATH),
        exist_ok=True
    )

    with open(
        OUTPUT_CSV_PATH,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "model",
            "baseline_accuracy",
            "latra_accuracy",
            "accuracy_drop",
            "baseline_loss",
            "latra_loss",
            "baseline_parameters",
            "latra_estimated_parameters",
            "parameter_reduction_percent",
            "average_reconstruction_error"
        ])

        writer.writerow([
            "LATRA",
            baseline_accuracy,
            latra_accuracy,
            accuracy_drop,
            baseline_loss,
            latra_loss,
            total_original_params,
            total_new_params,
            total_reduction,
            avg_error
        ])

    print()
    print("=" * 80)
    print("LATRA Full Model Evaluation Completed")
    print("=" * 80)

    print()
    print(
        f"Result saved to:"
        f"\n{OUTPUT_CSV_PATH}"
    )


if __name__ == "__main__":
    main()
