from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app.services import rag_service


@pytest.fixture(autouse=True)
def reset_rag_globals():
    rag_service._chunks = None
    yield
    rag_service._chunks = None


class TestSplitIntoChunks:
    def test_split_empty_text_returns_empty_list(self):
        assert rag_service._split_into_chunks("   ") == []

    def test_split_text_with_overlap(self):
        text = "a" * 600
        chunks = rag_service._split_into_chunks(text, size=500, overlap=50)
        assert len(chunks) >= 2
        assert chunks[0] == "a" * 500


class TestLoadKnowledgeBase:
    def test_load_knowledge_base_file_not_found(self):
        with patch.object(rag_service, "settings") as mock_settings:
            mock_settings.RAG_PDF_PATH = "/missing/knowledge_base.pdf"
            with pytest.raises(FileNotFoundError, match="RAG_PDF_PATH does not exist"):
                rag_service.load_knowledge_base()

    def test_load_knowledge_base_success(self, tmp_path):
        pdf_file = tmp_path / "kb.pdf"
        pdf_file.touch()

        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Dermatology acne treatment skin care " * 20
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with (
            patch.object(rag_service, "settings") as mock_settings,
            patch("app.services.rag_service.PdfReader", return_value=mock_reader),
        ):
            mock_settings.RAG_PDF_PATH = str(pdf_file)
            rag_service.load_knowledge_base()

        assert rag_service._chunks is not None
        assert len(rag_service._chunks) > 0

    def test_load_knowledge_base_handles_empty_page_text(self, tmp_path):
        pdf_file = tmp_path / "kb.pdf"
        pdf_file.touch()

        mock_page = MagicMock()
        mock_page.extract_text.return_value = None
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]

        with (
            patch.object(rag_service, "settings") as mock_settings,
            patch("app.services.rag_service.PdfReader", return_value=mock_reader),
        ):
            mock_settings.RAG_PDF_PATH = str(pdf_file)
            rag_service.load_knowledge_base()

        assert rag_service._chunks == []


class TestFindRelevantChunks:
    def test_find_relevant_chunks_returns_top_k(self):
        rag_service._chunks = [
            "acne treatment skin care",
            "unrelated nutrition advice",
            "acne hormonal therapy",
            "general wellness tips",
        ]

        result = rag_service.find_relevant_chunks("acne skin treatment", top_k=2)

        assert len(result) == 2
        assert "acne" in result[0].lower()

    def test_find_relevant_chunks_empty_base_returns_empty_list(self):
        rag_service._chunks = []
        assert rag_service.find_relevant_chunks("acne", top_k=3) == []

    def test_find_relevant_chunks_empty_query_returns_first_chunks(self):
        rag_service._chunks = ["chunk-a", "chunk-b", "chunk-c", "chunk-d"]
        assert rag_service.find_relevant_chunks("   ", top_k=2) == ["chunk-a", "chunk-b"]

    def test_find_relevant_chunks_no_matching_words_returns_prefix(self):
        rag_service._chunks = ["alpha", "beta", "gamma"]
        assert rag_service.find_relevant_chunks("x", top_k=2) == ["alpha", "beta"]

    def test_find_relevant_chunks_requires_loaded_base(self):
        with pytest.raises(RuntimeError, match="Knowledge base not loaded"):
            rag_service.find_relevant_chunks("acne")


class TestFormatQuestionnaire:
    def test_format_questionnaire_empty_dict(self):
        assert rag_service._format_questionnaire(None) == "Анкета не заповнена"
        assert rag_service._format_questionnaire({}) == "Анкета не заповнена"

    def test_format_questionnaire_skips_empty_values(self):
        text = rag_service._format_questionnaire(
            {"skin_type": "oily", "sleep_quality": "", "stress_trigger": None},
        )
        assert "Тип шкіри: oily" in text
        assert "Якість сну" not in text

    def test_format_questionnaire_all_values_empty_returns_default(self):
        text = rag_service._format_questionnaire({"skin_type": "", "sleep_quality": None})
        assert text == "Анкета не заповнена"


class TestGenerateRecommendation:
    def _mock_httpx_response(self, json_data: dict, status_code: int = 200) -> MagicMock:
        response = MagicMock()
        response.status_code = status_code
        response.json.return_value = json_data
        response.raise_for_status = MagicMock()
        return response

    def test_generate_recommendation_success(self):
        rag_service._chunks = ["acne treatment recommendations"]
        mock_response = self._mock_httpx_response(
            {"choices": [{"message": {"content": "  Рекомендації для догляду за шкірою.  "}}]},
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response

        with patch("app.services.rag_service.httpx.Client", return_value=mock_client):
            text, prompt = rag_service.generate_recommendation("acne1", 0.87, {"skin_type": "oily"})

        assert text == "Рекомендації для догляду за шкірою."
        assert "2 ступінь акне" in prompt
        assert "oily" in prompt

    def test_generate_recommendation_connect_error_fallback(self):
        rag_service._chunks = ["some medical context"]
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = httpx.ConnectError("connection refused")

        with patch("app.services.rag_service.httpx.Client", return_value=mock_client):
            text, prompt = rag_service.generate_recommendation("acne0", 0.75, None)

        assert text == rag_service.FALLBACK_LLM_MESSAGE
        assert "1 ступінь акне" in prompt

    def test_generate_recommendation_http_status_error_fallback(self):
        rag_service._chunks = ["context"]
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "server error",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response

        with patch("app.services.rag_service.httpx.Client", return_value=mock_client):
            text, _ = rag_service.generate_recommendation("acne2", 0.6, None)

        assert text == rag_service.FALLBACK_LLM_MESSAGE

    def test_generate_recommendation_missing_choices_fallback(self):
        rag_service._chunks = ["context"]
        mock_response = self._mock_httpx_response({"choices": []})
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response

        with patch("app.services.rag_service.httpx.Client", return_value=mock_client):
            text, _ = rag_service.generate_recommendation("acne3", 0.9, None)

        assert text == rag_service.FALLBACK_LLM_MESSAGE

    def test_generate_recommendation_empty_content_fallback(self):
        rag_service._chunks = ["context"]
        mock_response = self._mock_httpx_response({"choices": [{"message": {"content": "   "}}]})
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response

        with patch("app.services.rag_service.httpx.Client", return_value=mock_client):
            text, _ = rag_service.generate_recommendation("clear", 0.99, None)

        assert text == rag_service.FALLBACK_LLM_MESSAGE

    def test_generate_recommendation_unexpected_json_format_fallback(self):
        rag_service._chunks = ["context"]
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.side_effect = ValueError("invalid json")
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response

        with patch("app.services.rag_service.httpx.Client", return_value=mock_client):
            text, _ = rag_service.generate_recommendation("acne1", 0.5, None)

        assert text == rag_service.FALLBACK_LLM_MESSAGE

    def test_generate_recommendation_unknown_class_uses_raw_label(self):
        rag_service._chunks = ["general skin advice"]
        mock_response = self._mock_httpx_response(
            {"choices": [{"message": {"content": "Поради."}}]},
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response

        with patch("app.services.rag_service.httpx.Client", return_value=mock_client):
            text, prompt = rag_service.generate_recommendation("unknown_class", 0.4, None)

        assert text == "Поради."
        assert "unknown_class" in prompt

    def test_generate_recommendation_missing_message_content_key(self):
        rag_service._chunks = ["context"]
        mock_response = self._mock_httpx_response({"choices": [{"message": {}}]})
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_response

        with patch("app.services.rag_service.httpx.Client", return_value=mock_client):
            text, _ = rag_service.generate_recommendation("acne0", 0.8, None)

        assert text == rag_service.FALLBACK_LLM_MESSAGE
