from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx
from pypdf import PdfReader

from app.config import settings

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 500
_CHUNK_OVERLAP = 50

_chunks: list[str] | None = None

QUESTIONNAIRE_LABELS_UK: dict[str, str] = {
    "has_family_history": "Сімейна схильність",
    "has_hormonal_acne": "Гормональне акне",
    "uses_comedogenic_products": "Комедогенна косметика",
    "hair_products_trigger": "Засоби для волосся",
    "stress_trigger": "Стрес погіршує",
    "sleep_quality": "Якість сну",
    "is_smoking": "Паління",
    "water_intake": "Споживання води",
    "dairy_consumption": "Молочні продукти",
    "high_glycemic_diet": "Висококалорійна дієта",
    "food_triggers": "Харчові тригери",
    "skin_type": "Тип шкіри",
    "oily_skin": "Жирна шкіра",
    "jawline_acne": "Акне на підборідді",
    "has_demodex": "Демодекс",
    "vitamin_d_level": "Рівень вітаміну D",
    "dairy_sensitivity": "Чутливість до молочного",
}

ACNE_CLASS_LABELS_UK: dict[str, str] = {
    "acne0": "1 ступінь акне (легкий)",
    "acne1": "2 ступінь акне (помірний)",
    "acne2": "3 ступінь акне (помірно-важкий)",
    "acne3": "4 ступінь акне (важкий)",
    "clear": "чиста шкіра (акне відсутнє)",
}

FALLBACK_LLM_MESSAGE = (
    "Сервіс рекомендацій тимчасово недоступний. Спробуйте пізніше або зверніться до лікаря-дерматолога."
)


def _split_into_chunks(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def load_knowledge_base() -> None:
    global _chunks
    path = Path(settings.RAG_PDF_PATH)
    if not path.is_file():
        raise FileNotFoundError(f"RAG_PDF_PATH does not exist: {path.resolve()}")

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    full = "\n".join(parts)
    _chunks = _split_into_chunks(full)


def _ensure_chunks() -> list[str]:
    if _chunks is None:
        raise RuntimeError("Knowledge base not loaded; call load_knowledge_base() first.")
    return _chunks


def find_relevant_chunks(query: str, top_k: int = 3) -> list[str]:
    chunks = _ensure_chunks()
    if not chunks:
        return []
    if not query.strip():
        return chunks[:top_k]

    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
    if not words:
        return chunks[:top_k]

    scored: list[tuple[int, str]] = []
    for ch in chunks:
        cl = ch.lower()
        score = sum(1 for w in words if w in cl)
        scored.append((score, ch))
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    return [ch for _, ch in scored[:top_k]]


def _format_questionnaire(questionnaire: dict | None) -> str:
    if not questionnaire:
        return "Анкета не заповнена"
    lines: list[str] = []
    for key, label in QUESTIONNAIRE_LABELS_UK.items():
        if key not in questionnaire:
            continue
        val = questionnaire[key]
        if val is None or val == "":
            continue
        lines.append(f"- {label}: {val}")
    if not lines:
        return "Анкета не заповнена"
    return "\n".join(lines)


def _build_user_prompt(
    class_label: str,
    confidence: float,
    questionnaire: dict | None,
    relevant_chunks: list[str],
) -> str:
    q_text = _format_questionnaire(questionnaire)
    chunks_text = "\n\n".join(relevant_chunks) if relevant_chunks else "(немає відповідних уривків)"
    return f"""
Результат аналізу шкіри:
- Клас акне: {class_label}
- Впевненість моделі: {confidence:.0%}

Профіль користувача:
{q_text}

Релевантна медична інформація:
{chunks_text}

Надай структуровані рекомендації по догляду за шкірою, харчуванню та способу життя.
""".strip()


def generate_recommendation(
    predicted_class: str,
    confidence: float,
    questionnaire: dict | None,
) -> tuple[str, str]:
    """Returns (recommendation_text, user_prompt_used)."""
    class_label = ACNE_CLASS_LABELS_UK.get(predicted_class, predicted_class)
    query = f"{class_label} акне шкіра лікування догляд"
    if questionnaire:
        parts = [class_label] + [str(v) for v in questionnaire.values() if v]
        query = " ".join(parts)

    relevant = find_relevant_chunks(query, top_k=3)
    user_content = _build_user_prompt(class_label, confidence, questionnaire, relevant)

    payload: dict = {
        "model": settings.LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a dermatology assistant. Answer in Ukrainian. "
                    "Be specific and practical."
                ),
            },
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
    }

    try:
        with httpx.Client(timeout=300.0) as client:
            response = client.post(settings.LLM_API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
        choices = data.get("choices") or []
        if not choices:
            logger.warning("LLM response missing choices: %s", data)
            return FALLBACK_LLM_MESSAGE, user_content
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not content or not str(content).strip():
            return FALLBACK_LLM_MESSAGE, user_content
        return str(content).strip(), user_content
    except httpx.HTTPError as e:
        logger.warning("LLM HTTP error: %s", e)
        return FALLBACK_LLM_MESSAGE, user_content
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("LLM response parse error: %s", e)
        return FALLBACK_LLM_MESSAGE, user_content
