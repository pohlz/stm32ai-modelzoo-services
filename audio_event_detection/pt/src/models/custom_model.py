"""Custom KWS model factory; the upstream network returns unnormalized logits."""

from .bcresnet import BCResNets


def get_custom_model(input_shape=(1, 40, 96), num_classes=12, tau=1):
    """Build Qualcomm BC-ResNet without changing its architecture or state keys."""
    if len(input_shape) != 3 or tuple(input_shape[:2]) != (1, 40) or input_shape[2] < 1:
        raise ValueError("BC-ResNet expects channels-first input_shape [1, 40, time]")
    if tau not in (1, 1.5, 2, 3, 6, 8):
        raise ValueError("tau must be one of the upstream scales: 1, 1.5, 2, 3, 6, 8")
    if num_classes < 2:
        raise ValueError("At least two classes are required")
    return BCResNets(int(tau * 8), num_classes=num_classes)
