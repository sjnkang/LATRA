import os
import csv
import torch
import torch.nn as nn
import numpy as np


# ============================================================
# LATRA - Baseline Analysis
# ResNet-8 / CIFAR-10
#
# 1. Load Best Checkpoint
# 2. Model Statistics
#    - Parameters
#    - FP32 Model Size
#    - FLOPs
# 3. Layer-wise SVD Analysis
#    - Singular Values
#    - Energy
#    - Rank-90 / 95 / 99
#    - Reconstruction Error
# ============================================================


# ============================================================
# 1. ResNet-8 Architecture
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
# 2. Configuration
# ============================================================

CHECKPOINT_PATH = (
    "./checkpoints/resnet8_cifar10_best.pth"
)

OUTPUT_DIR = "./analysis"

SVD_CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "resnet8_layerwise_svd.csv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# CIFAR-10 input
INPUT_SIZE = 32


# ============================================================
# 3. Model Statistics
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

    return (
        param_size + buffer_size
    ) / (1024 ** 2)


# ============================================================
# 4. FLOPs Calculation
# ============================================================

def calculate_flops(model):

    """
    Calculate approximate FLOPs for a single
    CIFAR-10 input: [1, 3, 32, 32]

    Convention:
        One multiply + one add = 2 FLOPs
    """

    total_flops = 0

    hooks = []

    def conv_hook(module, inputs, output):

        nonlocal total_flops

        batch_size = output.shape[0]

        out_channels = output.shape[1]

        out_height = output.shape[2]
        out_width = output.shape[3]

        kernel_h = module.kernel_size[0]
        kernel_w = module.kernel_size[1]

        in_channels = module.in_channels

        groups = module.groups

        kernel_ops = (
            kernel_h
            * kernel_w
            * (in_channels // groups)
        )

        output_elements = (
            batch_size
            * out_channels
            * out_height
            * out_width
        )

        # Multiply + Add
        flops = (
            output_elements
            * kernel_ops
            * 2
        )

        # Bias addition
        if module.bias is not None:

            flops += output_elements

        total_flops += flops

    def linear_hook(module, inputs, output):

        nonlocal total_flops

        batch_size = output.shape[0]

        flops = (
            batch_size
            * module.in_features
            * module.out_features
            * 2
        )

        if module.bias is not None:

            flops += (
                batch_size
                * module.out_features
            )

        total_flops += flops

    for module in model.modules():

        if isinstance(
            module,
            nn.Conv2d
        ):

            hooks.append(
                module.register_forward_hook(
                    conv_hook
                )
            )

        elif isinstance(
            module,
            nn.Linear
        ):

            hooks.append(
                module.register_forward_hook(
                    linear_hook
                )
            )

    model.eval()

    dummy_input = torch.zeros(
        1,
        3,
        INPUT_SIZE,
        INPUT_SIZE,
        device=DEVICE
    )

    with torch.no_grad():

        model(dummy_input)

    for hook in hooks:

        hook.remove()

    return total_flops


# ============================================================
# 5. Layer-wise FLOPs
# ============================================================

def get_layer_flops(model):

    results = []

    hooks = []

    def make_hook(name):

        def hook(module, inputs, output):

            batch_size = output.shape[0]

            out_channels = output.shape[1]

            out_height = output.shape[2]
            out_width = output.shape[3]

            kernel_h = module.kernel_size[0]
            kernel_w = module.kernel_size[1]

            in_channels = module.in_channels

            groups = module.groups

            kernel_ops = (
                kernel_h
                * kernel_w
                * (in_channels // groups)
            )

            output_elements = (
                batch_size
                * out_channels
                * out_height
                * out_width
            )

            flops = (
                output_elements
                * kernel_ops
                * 2
            )

            if module.bias is not None:

                flops += output_elements

            results.append({

                "layer": name,

                "type": "Conv2d",

                "input_channels": in_channels,

                "output_channels": out_channels,

                "kernel": (
                    kernel_h,
                    kernel_w
                ),

                "stride": module.stride,

                "output_height": out_height,

                "output_width": out_width,

                "parameters": (
                    module.weight.numel()
                    + (
                        module.bias.numel()
                        if module.bias is not None
                        else 0
                    )
                ),

                "flops": flops
            })

        return hook

    for name, module in model.named_modules():

        if isinstance(
            module,
            nn.Conv2d
        ):

            hook = module.register_forward_hook(
                make_hook(name)
            )

            hooks.append(hook)

    model.eval()

    dummy_input = torch.zeros(
        1,
        3,
        INPUT_SIZE,
        INPUT_SIZE,
        device=DEVICE
    )

    with torch.no_grad():

        model(dummy_input)

    for hook in hooks:

        hook.remove()

    return results


# ============================================================
# 6. SVD Utilities
# ============================================================

def find_energy_rank(
    singular_values,
    threshold
):

    singular_values = (
        singular_values.cpu()
        .numpy()
    )

    energy = singular_values ** 2

    cumulative_energy = np.cumsum(
        energy
    )

    total_energy = energy.sum()

    ratio = (
        cumulative_energy
        / total_energy
    )

    rank = (
        np.searchsorted(
            ratio,
            threshold
        )
        + 1
    )

    return rank


def reconstruction_error(
    matrix,
    singular_values,
    rank
):

    """
    Relative Frobenius reconstruction error.

    ||W - W_r||_F / ||W||_F

    For truncated SVD:
        error^2 =
        sum(discarded singular values^2)
        /
        sum(all singular values^2)
    """

    s = singular_values

    total_energy = torch.sum(
        s ** 2
    )

    retained_energy = torch.sum(
        s[:rank] ** 2
    )

    error = torch.sqrt(
        torch.clamp(
            1.0
            - retained_energy
            / total_energy,
            min=0.0
        )
    )

    return error.item()


# ============================================================
# 7. Layer-wise SVD Analysis
# ============================================================

def analyze_layer_svd(
    model
):

    """
    Analyze only the 7 main-branch
    3x3 convolution layers.

    Shortcut 1x1 convolutions are
    intentionally excluded from the
    LATRA decomposition target.
    """

    target_layers = [

        "conv1",

        "layer1.conv1",
        "layer1.conv2",

        "layer2.conv1",
        "layer2.conv2",

        "layer3.conv1",
        "layer3.conv2"
    ]

    results = []

    print("\n")
    print("=" * 80)
    print("LATRA - Layer-wise SVD Analysis")
    print("=" * 80)

    for layer_name in target_layers:

        layer = model

        for part in layer_name.split("."):

            layer = getattr(
                layer,
                part
            )

        weight = (
            layer.weight
            .detach()
            .float()
            .cpu()
        )

        out_channels = weight.shape[0]

        in_channels = weight.shape[1]

        kernel_h = weight.shape[2]

        kernel_w = weight.shape[3]

        # ----------------------------------------------------
        # Conv Weight → 2D Matrix
        #
        # [Out, In, H, W]
        # →
        # [Out, In*H*W]
        # ----------------------------------------------------

        matrix = weight.reshape(
            out_channels,
            -1
        )

        matrix_rows = matrix.shape[0]

        matrix_cols = matrix.shape[1]

        max_rank = min(
            matrix_rows,
            matrix_cols
        )

        # ----------------------------------------------------
        # SVD
        # ----------------------------------------------------

        _, singular_values, _ = torch.linalg.svd(
            matrix,
            full_matrices=False
        )

        # ----------------------------------------------------
        # Energy Ranks
        # ----------------------------------------------------

        rank90 = find_energy_rank(
            singular_values,
            0.90
        )

        rank95 = find_energy_rank(
            singular_values,
            0.95
        )

        rank99 = find_energy_rank(
            singular_values,
            0.99
        )

        # ----------------------------------------------------
        # Reconstruction Error
        # ----------------------------------------------------

        error90 = reconstruction_error(
            matrix,
            singular_values,
            rank90
        )

        error95 = reconstruction_error(
            matrix,
            singular_values,
            rank95
        )

        error99 = reconstruction_error(
            matrix,
            singular_values,
            rank99
        )

        # ----------------------------------------------------
        # Compression ratio based on
        # simple rank-r matrix factorization
        #
        # W ≈ U_r @ V_r
        #
        # parameters:
        #   r * (m + n)
        # ----------------------------------------------------

        original_params = (
            matrix_rows
            * matrix_cols
        )

        compressed_params_90 = (
            rank90
            * (
                matrix_rows
                + matrix_cols
            )
        )

        compressed_params_95 = (
            rank95
            * (
                matrix_rows
                + matrix_cols
            )
        )

        compressed_params_99 = (
            rank99
            * (
                matrix_rows
                + matrix_cols
            )
        )

        compression_90 = (
            original_params
            / compressed_params_90
        )

        compression_95 = (
            original_params
            / compressed_params_95
        )

        compression_99 = (
            original_params
            / compressed_params_99
        )

        result = {

            "layer": layer_name,

            "shape": str(
                tuple(weight.shape)
            ),

            "matrix_rows": matrix_rows,

            "matrix_cols": matrix_cols,

            "max_rank": max_rank,

            "rank90": rank90,

            "rank95": rank95,

            "rank99": rank99,

            "error90": error90,

            "error95": error95,

            "error99": error99,

            "compression90": compression_90,

            "compression95": compression_95,

            "compression99": compression_99
        }

        results.append(result)

        # ----------------------------------------------------
        # Print
        # ----------------------------------------------------

        print("\nLayer:", layer_name)

        print(
            "Weight Shape:",
            tuple(weight.shape)
        )

        print(
            "Matrix Shape:",
            f"{matrix_rows} x {matrix_cols}"
        )

        print(
            "Max Rank:",
            max_rank
        )

        print(
            f"Rank-90%: {rank90:>3} "
            f"| Error: {error90:.6f}"
        )

        print(
            f"Rank-95%: {rank95:>3} "
            f"| Error: {error95:.6f}"
        )

        print(
            f"Rank-99%: {rank99:>3} "
            f"| Error: {error99:.6f}"
        )

        print(
            f"Compression @90%: "
            f"{compression_90:.2f}x"
        )

        print(
            f"Compression @95%: "
            f"{compression_95:.2f}x"
        )

        print(
            f"Compression @99%: "
            f"{compression_99:.2f}x"
        )

    return results


# ============================================================
# 8. Save SVD Results
# ============================================================

def save_svd_results(
    results,
    path
):

    fieldnames = [

        "layer",

        "shape",

        "matrix_rows",
        "matrix_cols",

        "max_rank",

        "rank90",
        "rank95",
        "rank99",

        "error90",
        "error95",
        "error99",

        "compression90",
        "compression95",
        "compression99"
    ]

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            results
        )


# ============================================================
# 9. Main
# ============================================================

def main():

    print("=" * 80)
    print("LATRA - ResNet-8 Baseline Analysis")
    print("=" * 80)

    print(
        f"\nDevice: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ========================================================
    # Load Model
    # ========================================================

    print("\n" + "=" * 80)
    print("Loading Best Checkpoint")
    print("=" * 80)

    model = ResNet8(
        num_classes=10
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "Best Epoch:",
        checkpoint["epoch"]
    )

    print(
        "Validation Accuracy:",
        f"{checkpoint['val_accuracy']:.2f}%"
    )

    # ========================================================
    # Model Statistics
    # ========================================================

    print("\n" + "=" * 80)
    print("Original ResNet-8 Baseline")
    print("=" * 80)

    parameters = count_parameters(
        model
    )

    model_size = get_model_size_mb(
        model
    )

    print(
        f"Parameters: {parameters:,}"
    )

    print(
        f"FP32 Model Size: "
        f"{model_size:.4f} MB"
    )

    # ========================================================
    # FLOPs
    # ========================================================

    print("\n" + "=" * 80)
    print("FLOPs Analysis")
    print("=" * 80)

    total_flops = calculate_flops(
        model
    )

    print(
        f"Total FLOPs: "
        f"{total_flops:,}"
    )

    print(
        f"Total MFLOPs: "
        f"{total_flops / 1e6:.3f}"
    )

    # ========================================================
    # Layer-wise FLOPs
    # ========================================================

    print("\n" + "=" * 80)
    print("Layer-wise FLOPs")
    print("=" * 80)

    layer_flops = get_layer_flops(
        model
    )

    for item in layer_flops:

        print(
            f"{item['layer']:25s} | "
            f"Params: "
            f"{item['parameters']:7,d} | "
            f"FLOPs: "
            f"{item['flops'] / 1e6:8.3f} MFLOPs"
        )

    # ========================================================
    # SVD
    # ========================================================

    svd_results = analyze_layer_svd(
        model
    )

    # ========================================================
    # Save
    # ========================================================

    save_svd_results(
        svd_results,
        SVD_CSV_PATH
    )

    print("\n" + "=" * 80)
    print("Analysis Completed")
    print("=" * 80)

    print(
        "\nSVD results saved to:"
    )

    print(
        SVD_CSV_PATH
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()
