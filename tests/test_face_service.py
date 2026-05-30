from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.services import face_service


@pytest.fixture(autouse=True)
def reset_detector():
    face_service._detector = None
    yield
    face_service._detector = None


@pytest.fixture
def sample_image() -> Image.Image:
    return Image.new("RGB", (100, 100))


class TestLoadDetector:
    def test_load_detector_initializes_mtcnn(self):
        mock_mtcnn = MagicMock()
        with patch("app.services.face_service.MTCNN", return_value=mock_mtcnn) as mock_cls:
            face_service.load_detector()

        mock_cls.assert_called_once_with(keep_all=True)
        assert face_service._detector is mock_mtcnn


class TestIsHumanFace:
    def test_is_human_face_true_when_faces_detected(self, sample_image: Image.Image):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([[0, 0, 50, 50]], None)
        face_service._detector = mock_detector

        assert face_service.is_human_face(sample_image) is True
        mock_detector.detect.assert_called_once_with(sample_image)

    def test_is_human_face_false_when_no_faces(self, sample_image: Image.Image):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = (None, None)
        face_service._detector = mock_detector

        assert face_service.is_human_face(sample_image) is False

    def test_is_human_face_false_when_empty_boxes(self, sample_image: Image.Image):
        mock_detector = MagicMock()
        mock_detector.detect.return_value = ([], None)
        face_service._detector = mock_detector

        assert face_service.is_human_face(sample_image) is False

    def test_is_human_face_false_on_detector_exception(self, sample_image: Image.Image):
        mock_detector = MagicMock()
        mock_detector.detect.side_effect = RuntimeError("MTCNN failure")
        face_service._detector = mock_detector

        assert face_service.is_human_face(sample_image) is False

    def test_is_human_face_false_when_detector_not_loaded(self, sample_image: Image.Image):
        assert face_service._detector is None
        assert face_service.is_human_face(sample_image) is False
