"""
Direct patch for e3nn Linear forward method to handle missing _compiled_main.

This is a more targeted fix that directly addresses the missing _compiled_main issue
without trying to fix the complex code generation system.
"""

import warnings
from typing import Optional

import torch


def patch_e3nn_linear_forward():
    """
    Patch e3nn Linear.forward to handle missing _compiled_main method.

    This is a targeted fix for the specific issue where MACE models serialized
    with e3nn 0.4.4 fail to load with e3nn 0.5+ due to missing compiled methods.
    """
    try:
        import e3nn.o3._linear as linear_module
    except ImportError:
        warnings.warn("e3nn not available, skipping Linear forward patch")
        return False

    # Check if already patched
    if hasattr(linear_module.Linear.forward, "_qme_patched"):
        return True

    # Store original method
    original_forward = linear_module.Linear.forward

    def patched_forward(
        self,
        features: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ):
        """
        Patched forward method that handles missing _compiled_main.

        If _compiled_main is available, use it. Otherwise, fall back to a
        manual implementation that replicates the e3nn Linear logic.
        """
        # Try to use compiled method if available
        if hasattr(self, "_compiled_main"):
            return self._compiled_main(features, weight, bias)

        # Fallback implementation
        try:
            # Use the weights from the module if not provided
            if weight is None:
                weight = self.weight
            if bias is None and hasattr(self, "bias") and self.bias is not None:
                bias = self.bias

            # Manual implementation of e3nn Linear logic
            # This is based on the e3nn 0.4.4 implementation

            # Get input and output irreps
            irreps_in = self.irreps_in
            irreps_out = self.irreps_out

            # Reshape features for processing
            batch_shape = features.shape[:-1]
            features = features.reshape(-1, irreps_in.dim)

            # Apply linear transformation
            output = torch.zeros(
                features.shape[0],
                irreps_out.dim,
                dtype=features.dtype,
                device=features.device,
            )

            # Process each irrep pair
            weight_idx = 0
            for i, (mul_out, ir_out) in enumerate(irreps_out):
                for j, (mul_in, ir_in) in enumerate(irreps_in):
                    if ir_in == ir_out:
                        # Extract relevant weights
                        w = weight[weight_idx : weight_idx + mul_out * mul_in].reshape(
                            mul_out, mul_in
                        )
                        weight_idx += mul_out * mul_in

                        # Get input features for this irrep
                        start_in = irreps_in[:j].dim
                        end_in = start_in + mul_in * ir_in.dim
                        feat_in = features[:, start_in:end_in].reshape(
                            -1, mul_in, ir_in.dim
                        )

                        # Apply transformation
                        feat_out = torch.einsum("bmi,om->boi", feat_in, w)

                        # Place in output
                        start_out = irreps_out[:i].dim
                        end_out = start_out + mul_out * ir_out.dim
                        output[:, start_out:end_out] = feat_out.reshape(
                            -1, mul_out * ir_out.dim
                        )

            # Add bias if present
            if bias is not None:
                output = output + bias

            # Reshape back to original batch shape
            output = output.reshape(*batch_shape, irreps_out.dim)

            return output

        except Exception as e:
            # If our fallback fails, try the original method
            warnings.warn(f"E3NN Linear fallback failed: {e}, trying original method")
            return original_forward(self, features, weight, bias)

    # Mark as patched and apply
    patched_forward._qme_patched = True
    linear_module.Linear.forward = patched_forward

    return True


def unpatch_e3nn_linear_forward():
    """
    Remove the e3nn Linear forward patch and restore original behavior.
    """
    try:
        import e3nn.o3._linear as linear_module
    except ImportError:
        return False

    # Check if patched
    if not hasattr(linear_module.Linear.forward, "_qme_patched"):
        return False

    # We can't easily restore the original, so we'll leave the patch in place
    warnings.warn("E3NN Linear patch cannot be safely removed once applied")
    return False


class E3NNLinearPatchContext:
    """
    Context manager for applying e3nn Linear patches.

    Usage:
        with E3NNLinearPatchContext():
            # Load MACE models here
            calc = MACECalculator(...)
    """

    def __init__(self):
        self.patched = False

    def __enter__(self):
        self.patched = patch_e3nn_linear_forward()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't unpatch - it's safer to leave the compatible version in place
        pass
