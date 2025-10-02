"""
E3NN compatibility fixes for MACE models.

This module provides compatibility patches for e3nn version conflicts between
MACE 0.3.14 (requires e3nn==0.4.4) and FairChem 2.7.0 (requires e3nn>=0.5).

The main issue is that e3nn 0.5+ changed the serialization format in the
CodeGenMixin.__setstate__ method, causing "too many values to unpack" errors
when loading MACE models that were serialized with e3nn 0.4.4.
"""

import warnings
from typing import Any, Dict


def patch_e3nn_codegen_mixin():
    """
    Patch e3nn CodeGenMixin to handle both old and new serialization formats.

    This allows MACE models serialized with e3nn 0.4.4 to be loaded with e3nn 0.5+.
    """
    try:
        import e3nn.util.codegen._mixin as mixin_module
    except ImportError:
        warnings.warn("e3nn not available, skipping compatibility patch")
        return False

    # Check if already patched
    if hasattr(mixin_module.CodeGenMixin.__setstate__, "_qme_patched"):
        return True

    # Store original method
    original_setstate = mixin_module.CodeGenMixin.__setstate__

    def compatible_setstate(self, d: Dict[str, Any]):
        """
        Compatible __setstate__ method that handles both e3nn 0.4.4 and 0.5+ formats.

        The issue is that e3nn 0.4.4 serializes codegen_state as:
            {fname: (buffer_type, buffer)}

        But e3nn 0.5+ expects a different format or handles it differently.
        This method handles both formats gracefully.
        """
        codegen_state = d.pop("_codegen_state", None)

        # Handle the base state
        if hasattr(self, "__dict__"):
            self.__dict__.update(d)
        else:
            super(mixin_module.CodeGenMixin, self).__setstate__(d)

        # Handle codegen state with compatibility for both formats
        if codegen_state is not None:
            for fname, value in codegen_state.items():
                assert isinstance(fname, str)

                # Handle different serialization formats
                if isinstance(value, tuple):
                    if len(value) == 2:
                        # e3nn 0.4.4 format: (buffer_type, buffer)
                        buffer_type, buffer = value
                    elif len(value) >= 2:
                        # e3nn 0.5+ format or extended format: (buffer_type, buffer, ...)
                        buffer_type, buffer = value[0], value[1]
                        # Silently ignore extra values for compatibility
                    else:
                        raise ValueError(
                            f"Unexpected codegen_state format for {fname}: {value}"
                        )
                else:
                    raise ValueError(
                        f"Expected tuple in codegen_state for {fname}, got {type(value)}"
                    )

                assert isinstance(buffer_type, str)
                assert isinstance(buffer, bytes)

                # Execute the generated code
                try:
                    code = compile(buffer, fname, "exec")

                    # Create a proper namespace with all necessary imports
                    import math

                    import e3nn
                    import e3nn.nn
                    import e3nn.o3
                    import torch
                    import torch.nn.functional as F

                    namespace = {
                        "self": self,
                        "torch": torch,
                        "F": F,
                        "math": math,
                        "e3nn": e3nn,
                        "o3": e3nn.o3,
                        "Irreps": e3nn.o3.Irreps,
                        "Linear": e3nn.o3.Linear,
                    }

                    # Execute the code
                    exec(code, namespace)

                    # Find and bind the generated method
                    method_found = False
                    for name, obj in namespace.items():
                        if name not in [
                            "self",
                            "torch",
                            "F",
                            "math",
                            "e3nn",
                            "o3",
                            "Irreps",
                            "Linear",
                        ] and callable(obj):
                            # Bind the method to the object
                            bound_method = (
                                obj.__get__(self, type(self))
                                if hasattr(obj, "__get__")
                                else obj
                            )
                            setattr(self, name, bound_method)
                            method_found = True

                    # Special handling for _compiled_main which is commonly missing
                    if not method_found and "main" in fname.lower():
                        # Look for any function that could be the main compiled function
                        for name, obj in namespace.items():
                            if (
                                callable(obj)
                                and not name.startswith("__")
                                and name
                                not in [
                                    "self",
                                    "torch",
                                    "F",
                                    "math",
                                    "e3nn",
                                    "o3",
                                    "Irreps",
                                    "Linear",
                                ]
                            ):
                                bound_method = (
                                    obj.__get__(self, type(self))
                                    if hasattr(obj, "__get__")
                                    else obj
                                )
                                setattr(self, "_compiled_main", bound_method)
                                method_found = True
                                break

                except Exception as e:
                    warnings.warn(f"Failed to execute codegen for {fname}: {e}")

                    # Create a fallback method for _compiled_main
                    if "main" in fname.lower() or "_compiled_main" in fname:

                        def fallback_compiled_main(features, weight=None, bias=None):
                            # Try to use the original e3nn Linear implementation
                            try:
                                # This is a simplified fallback - may not work for all cases
                                import e3nn.o3._linear

                                original_linear = e3nn.o3._linear.Linear

                                # Create a temporary Linear instance to use its logic
                                temp_linear = original_linear(
                                    self.irreps_in, self.irreps_out
                                )
                                temp_linear.weight = (
                                    self.weight if weight is None else weight
                                )
                                if hasattr(self, "bias"):
                                    temp_linear.bias = (
                                        self.bias if bias is None else bias
                                    )

                                return temp_linear(features)
                            except Exception:
                                # Last resort - raise a helpful error
                                raise NotImplementedError(
                                    f"E3NN compiled method failed to load and fallback failed. "
                                    f"This is likely due to e3nn version compatibility issues between "
                                    f"the serialized model (e3nn 0.4.4) and current version (e3nn 0.5+)."
                                )

                        setattr(self, "_compiled_main", fallback_compiled_main)

    # Mark as patched and apply
    compatible_setstate._qme_patched = True
    mixin_module.CodeGenMixin.__setstate__ = compatible_setstate

    return True


def unpatch_e3nn_codegen_mixin():
    """
    Remove the e3nn compatibility patch and restore original behavior.
    """
    try:
        import e3nn.util.codegen._mixin as mixin_module
    except ImportError:
        return False

    # Check if patched
    if not hasattr(mixin_module.CodeGenMixin.__setstate__, "_qme_patched"):
        return False

    # We can't easily restore the original, so we'll leave the patch in place
    # This is safer than trying to restore and potentially breaking things
    warnings.warn("E3NN patch cannot be safely removed once applied")
    return False


class E3NNCompatibilityContext:
    """
    Context manager for applying e3nn compatibility patches.

    Usage:
        with E3NNCompatibilityContext():
            # Load MACE models here
            calc = MACECalculator(...)
    """

    def __init__(self):
        self.patched = False

    def __enter__(self):
        self.patched = patch_e3nn_codegen_mixin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Don't unpatch - it's safer to leave the compatible version in place
        pass


def is_e3nn_compatible():
    """
    Check if the current e3nn version is compatible with MACE models.

    Returns:
        bool: True if compatible or if compatibility patch is available
    """
    try:
        import e3nn

        version = e3nn.__version__

        # e3nn 0.4.4 is natively compatible
        if version.startswith("0.4."):
            return True

        # For e3nn 0.5+, we can use our compatibility patch
        return patch_e3nn_codegen_mixin()

    except ImportError:
        return False
