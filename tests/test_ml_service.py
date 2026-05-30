from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from app.services import ml_service


@pytest.fixture(autouse=True)
def reset_ml_globals():
    ml_service._device = None
    ml_service._model = None
    ml_service._prototypes = None
    ml_service._class_names = ()
    yield
    ml_service._device = None
    ml_service._model = None
    ml_service._prototypes = None
    ml_service._class_names = ()


def _set_loaded_model(
    embeddings: torch.Tensor | None = None,
    prototypes: torch.Tensor | None = None,
    class_names: tuple[str, ...] = ("acne0", "acne1"),
) -> MagicMock:
    mock_model = MagicMock()
    mock_model.return_value = embeddings or torch.tensor([[1.0, 0.0]])
    ml_service._model = mock_model
    ml_service._device = torch.device("cpu")
    ml_service._prototypes = prototypes or torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ml_service._class_names = class_names
    return mock_model


class TestLoadModel:
    def test_load_model_success_with_model_state_key(self, tmp_path: Path):
        model_file = tmp_path / "model.pt"
        model_file.touch()
        mock_net = MagicMock()

        with (
            patch("app.services.ml_service.EmbeddingNet", return_value=mock_net),
            patch("app.services.ml_service.torch.load", return_value={"model_state": {"layer": "w"}}) as mock_load,
            patch.object(ml_service, "settings") as mock_settings,
        ):
            mock_settings.MODEL_PATH = str(model_file)
            ml_service.load_model()

        mock_load.assert_called_once()
        mock_net.load_state_dict.assert_called_once_with({"layer": "w"})
        mock_net.to.assert_called_once()
        mock_net.eval.assert_called_once()
        assert ml_service._model is mock_net
        assert ml_service._device is not None

    def test_load_model_success_with_raw_state_dict(self, tmp_path: Path):
        model_file = tmp_path / "model.pt"
        model_file.touch()
        mock_net = MagicMock()
        state = {"encoder.weight": torch.zeros(1)}

        with (
            patch("app.services.ml_service.EmbeddingNet", return_value=mock_net),
            patch("app.services.ml_service.torch.load", return_value=state),
            patch.object(ml_service, "settings") as mock_settings,
        ):
            mock_settings.MODEL_PATH = str(model_file)
            ml_service.load_model()

        mock_net.load_state_dict.assert_called_once_with(state)

    def test_load_model_file_not_found(self):
        with patch.object(ml_service, "settings") as mock_settings:
            mock_settings.MODEL_PATH = "/nonexistent/missing_model.pt"
            with pytest.raises(FileNotFoundError, match="MODEL_PATH does not exist"):
                ml_service.load_model()

    def test_load_model_torch_load_typeerror_fallback(self, tmp_path: Path):
        model_file = tmp_path / "model.pt"
        model_file.touch()
        mock_net = MagicMock()

        with (
            patch("app.services.ml_service.EmbeddingNet", return_value=mock_net),
            patch(
                "app.services.ml_service.torch.load",
                side_effect=[TypeError("weights_only unsupported"), {"model_state": {}}],
            ) as mock_load,
            patch.object(ml_service, "settings") as mock_settings,
        ):
            mock_settings.MODEL_PATH = str(model_file)
            ml_service.load_model()

        assert mock_load.call_count == 2


class TestBuildPrototypes:
    def test_build_prototypes_missing_support_folder(self):
        _set_loaded_model()

        with patch.object(ml_service, "settings") as mock_settings:
            mock_settings.SUPPORT_SET_PATH = "/missing/support_set"
            with pytest.raises(FileNotFoundError, match="SUPPORT_SET_PATH is not a directory"):
                ml_service.build_prototypes()

    def test_build_prototypes_no_class_subfolders(self, tmp_path: Path):
        _set_loaded_model()
        support_dir = tmp_path / "support"
        support_dir.mkdir()

        with patch.object(ml_service, "settings") as mock_settings:
            mock_settings.SUPPORT_SET_PATH = str(support_dir)
            with pytest.raises(FileNotFoundError, match="No class subfolders"):
                ml_service.build_prototypes()

    def test_build_prototypes_no_images_in_folder(self, tmp_path: Path):
        _set_loaded_model()
        support_dir = tmp_path / "support"
        class_dir = support_dir / "acne0"
        class_dir.mkdir(parents=True)

        with patch.object(ml_service, "settings") as mock_settings:
            mock_settings.SUPPORT_SET_PATH = str(support_dir)
            with pytest.raises(FileNotFoundError, match="No images found"):
                ml_service.build_prototypes()

    def test_build_prototypes_success(self, tmp_path: Path):
        mock_model = _set_loaded_model()
        support_dir = tmp_path / "support"
        class_dir = support_dir / "acne0"
        class_dir.mkdir(parents=True)
        img_path = class_dir / "sample.jpg"
        img_path.touch()

        mock_pil = MagicMock()
        mock_pil.convert.return_value = Image.new("RGB", (10, 10))

        with (
            patch.object(ml_service, "settings") as mock_settings,
            patch("app.services.ml_service.Image.open") as mock_open,
            patch("app.services.ml_service.val_transform") as mock_transform,
        ):
            mock_settings.SUPPORT_SET_PATH = str(support_dir)
            mock_open.return_value.__enter__.return_value = mock_pil
            mock_transform.return_value = torch.zeros(3, 224, 224)
            mock_model.return_value = torch.tensor([[1.0, 0.0, 0.0]])

            ml_service.build_prototypes()

        assert ml_service._class_names == ("acne0",)
        assert ml_service._prototypes is not None
        assert ml_service._prototypes.shape[0] == 1

    def test_build_prototypes_requires_loaded_model(self):
        with patch.object(ml_service, "settings") as mock_settings:
            mock_settings.SUPPORT_SET_PATH = "/any/path"
            with pytest.raises(RuntimeError, match="Model not loaded"):
                ml_service.build_prototypes()


class TestPredict:
    def test_predict_returns_class_and_confidence_in_range(self):
        _set_loaded_model()
        image = Image.new("RGB", (64, 64))

        with patch("app.services.ml_service.val_transform") as mock_transform:
            mock_transform.return_value = torch.zeros(3, 224, 224)
            result = ml_service.predict(image)

        assert "predicted_class" in result
        assert "confidence" in result
        assert result["predicted_class"] == "acne0"
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_converts_non_rgb_image(self):
        _set_loaded_model()
        image = Image.new("L", (64, 64))

        with patch("app.services.ml_service.val_transform") as mock_transform:
            mock_transform.return_value = torch.zeros(3, 224, 224)
            result = ml_service.predict(image)

        assert result["predicted_class"] in ml_service._class_names

    def test_predict_requires_loaded_model(self):
        image = Image.new("RGB", (64, 64))
        with pytest.raises(RuntimeError, match="Model not loaded"):
            ml_service.predict(image)

    def test_predict_requires_prototypes(self):
        ml_service._model = MagicMock()
        ml_service._device = torch.device("cpu")
        ml_service._prototypes = None

        with pytest.raises(RuntimeError, match="Prototypes not built"):
            ml_service.predict(Image.new("RGB", (64, 64)))

    def test_predict_requires_class_names(self):
        mock_model = MagicMock()
        ml_service._model = mock_model
        ml_service._device = torch.device("cpu")
        ml_service._prototypes = torch.tensor([[1.0, 0.0]])
        ml_service._class_names = ()

        with pytest.raises(RuntimeError, match="Class names not set"):
            ml_service.predict(Image.new("RGB", (64, 64)))

    def test_predict_model_forward_failure_propagates(self):
        mock_model = _set_loaded_model()
        mock_model.side_effect = RuntimeError("invalid tensor shape")

        with (
            patch("app.services.ml_service.val_transform", return_value=torch.zeros(3, 224, 224)),
            pytest.raises(RuntimeError, match="invalid tensor shape"),
        ):
            ml_service.predict(Image.new("RGB", (64, 64)))


class TestEmbeddingNet:
    def test_embedding_net_resnet34_typeerror_fallback(self):
        mock_base = MagicMock()
        mock_base.children.return_value = [MagicMock()]

        with (
            patch("app.services.ml_service.models.resnet34", side_effect=[TypeError(), mock_base]),
            patch("app.services.ml_service.nn.Sequential"),
            patch("app.services.ml_service.nn.Flatten"),
            patch("app.services.ml_service.nn.Linear"),
            patch("app.services.ml_service.nn.BatchNorm1d"),
            patch("app.services.ml_service.nn.ReLU"),
            patch("app.services.ml_service.nn.Dropout"),
        ):
            net = ml_service.EmbeddingNet()
            assert net is not None


class TestInitializeForInference:
    def test_initialize_for_inference_calls_both_steps(self):
        with (
            patch("app.services.ml_service.load_model") as mock_load,
            patch("app.services.ml_service.build_prototypes") as mock_build,
        ):
            ml_service.initialize_for_inference()

        mock_load.assert_called_once()
        mock_build.assert_called_once()


class TestResolveDevice:
    def test_resolve_device_cpu_when_cuda_unavailable(self):
        with patch("app.services.ml_service.torch.cuda.is_available", return_value=False):
            device = ml_service._resolve_device()
        assert device.type == "cpu"

    def test_resolve_device_cuda_when_available(self):
        with patch("app.services.ml_service.torch.cuda.is_available", return_value=True):
            device = ml_service._resolve_device()
        assert device.type == "cuda"
