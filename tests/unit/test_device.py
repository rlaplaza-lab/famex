from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qme.utils.device import get_device_info, get_optimal_device, print_device_info, validate_device


class TestDeviceUtilities:
    def test_get_optimal_device_explicit(self):
        assert get_optimal_device("cpu") == "cpu"
        assert get_optimal_device("CUDA") == "cuda"  # Should lowercase
        assert get_optimal_device("cuda") == "cuda"

    @patch("qme.backends.dependencies.deps")
    def test_get_optimal_device_auto_detect_cuda(self, mock_deps):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_deps.has.return_value = True
        mock_deps.get.return_value = mock_torch

        device = get_optimal_device(None)

        assert device == "cuda"
        mock_deps.has.assert_called_with("torch")
        mock_deps.get.assert_called_with("torch")

    @patch("qme.backends.dependencies.deps")
    def test_get_optimal_device_auto_detect_cpu(self, mock_deps):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_deps.has.return_value = True
        mock_deps.get.return_value = mock_torch

        device = get_optimal_device(None)

        assert device == "cpu"

    @patch("qme.backends.dependencies.deps")
    def test_get_optimal_device_no_torch(self, mock_deps):
        mock_deps.has.return_value = False

        device = get_optimal_device(None)

        assert device == "cpu"

    @patch("qme.backends.dependencies.deps")
    def test_get_optimal_device_import_error(self, mock_deps):
        mock_deps.has.side_effect = ImportError("No module named torch")

        device = get_optimal_device(None)

        assert device == "cpu"

    def test_validate_device_none(self):
        with patch("qme.utils.device.get_optimal_device", return_value="cpu"):
            device = validate_device(None)
            assert device == "cpu"

    def test_validate_device_cpu(self):
        device = validate_device("cpu")
        assert device == "cpu"

    def test_validate_device_cuda(self):
        with patch("qme.backends.dependencies.deps") as mock_deps:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_deps.has.return_value = True
            mock_deps.get.return_value = mock_torch

            device = validate_device("cuda")
            assert device == "cuda"

    def test_validate_device_gpu_alias(self):
        with patch("qme.backends.dependencies.deps") as mock_deps:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            mock_deps.has.return_value = True
            mock_deps.get.return_value = mock_torch

            device = validate_device("gpu")
            assert device == "cuda"

    def test_validate_device_invalid(self):
        with pytest.raises(ValueError, match="Invalid device"):
            validate_device("invalid_device")

    def test_validate_device_cuda_not_available(self):
        with patch("qme.backends.dependencies.deps") as mock_deps:
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = False
            mock_deps.has.return_value = True
            mock_deps.get.return_value = mock_torch

            with pytest.raises(ValueError, match="CUDA device requested but CUDA is not available"):
                validate_device("cuda")

    def test_validate_device_cuda_no_torch(self):
        with patch("qme.backends.dependencies.deps") as mock_deps:
            mock_deps.has.return_value = False

            with pytest.raises(ValueError, match="PyTorch not available"):
                validate_device("cuda")

    @patch("qme.backends.dependencies.deps")
    def test_get_device_info_cpu(self, mock_deps):
        info = get_device_info("cpu")

        assert info["device"] == "cpu"
        assert info["cuda_available"] is False
        assert info["gpu_name"] is None
        assert info["gpu_memory"] is None

    @patch("qme.backends.dependencies.deps")
    def test_get_device_info_cuda_available(self, mock_deps):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_device_props = MagicMock()
        mock_device_props.total_memory = 8 * 1024**3  # 8 GB
        mock_torch.cuda.get_device_properties.return_value = mock_device_props
        mock_deps.has.return_value = True
        mock_deps.get.return_value = mock_torch

        info = get_device_info("cuda")

        assert info["device"] == "cuda"
        assert info["cuda_available"] is True
        assert info["gpu_name"] == "Test GPU"
        assert info["gpu_memory"] == 8 * 1024**3

    @patch("qme.backends.dependencies.deps")
    def test_get_device_info_cuda_not_available(self, mock_deps):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_deps.has.return_value = True
        mock_deps.get.return_value = mock_torch

        info = get_device_info("cuda")

        assert info["device"] == "cuda"
        assert info["cuda_available"] is False
        assert info["gpu_name"] is None
        assert info["gpu_memory"] is None

    @patch("qme.backends.dependencies.deps")
    def test_get_device_info_import_error(self, mock_deps):
        mock_deps.has.side_effect = ImportError("No module")

        info = get_device_info("cuda")

        assert info["device"] == "cuda"
        assert info["cuda_available"] is False
        assert info["gpu_name"] is None

    @patch("qme.utils.device.logger")
    @patch("qme.backends.dependencies.deps")
    def test_print_device_info_cpu(self, mock_deps, mock_logger):
        print_device_info("cpu")

        mock_logger.info.assert_called_with("💻 Using CPU device")

    @patch("qme.utils.device.logger")
    @patch("qme.backends.dependencies.deps")
    def test_print_device_info_cuda_available(
        self,
        mock_deps,
        mock_logger,
    ):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_deps.has.return_value = True
        mock_deps.get.return_value = mock_torch

        print_device_info("cuda")

        # Should log GPU name
        mock_logger.info.assert_called()
        call_args = str(mock_logger.info.call_args)
        assert "CUDA" in call_args or "Test GPU" in call_args

    @patch("qme.utils.device.logger")
    @patch("qme.backends.dependencies.deps")
    def test_print_device_info_cuda_not_available(
        self,
        mock_deps,
        mock_logger,
    ):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_deps.has.return_value = True
        mock_deps.get.return_value = mock_torch

        print_device_info("cuda")

        # Should log warning about CUDA not available
        mock_logger.warning.assert_called()
        call_args = str(mock_logger.warning.call_args)
        assert "CUDA" in call_args or "not available" in call_args

    @patch("qme.utils.device.logger")
    @patch("qme.backends.dependencies.deps")
    def test_print_device_info_no_torch(
        self,
        mock_deps,
        mock_logger,
    ):
        mock_deps.has.return_value = False

        print_device_info("cuda")

        # Should log warning about PyTorch not available
        mock_logger.warning.assert_called()
        call_args = str(mock_logger.warning.call_args)
        assert "PyTorch" in call_args or "not available" in call_args

    @patch("qme.utils.device.logger")
    @patch("qme.backends.dependencies.deps")
    def test_print_device_info_import_error(
        self,
        mock_deps,
        mock_logger,
    ):
        mock_deps.has.side_effect = ImportError("No module")

        # Should not raise exception
        print_device_info("cuda")

        # Should log warning
        mock_logger.warning.assert_called()
