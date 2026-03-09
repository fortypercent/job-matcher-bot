"""
embedding_pipeline.py — векторизация резюме и вакансий + матчинг

СТРАТЕГИЯ (OOM-fix):
L6 модель заблокирована на HuggingFace (401), поэтому оставляем L12.
Вместо этого убираем PyTorch и используем ONNX Runtime:
  - PyTorch runtime: ~200MB RAM → ONNX Runtime: ~30MB RAM
  - L12 модель ONNX: ~280MB в памяти
  - Итого: ~350MB вместо ~700MB → помещается в 512MB Railway

Модель: paraphrase-multilingual-MiniLM-L12-v2 (ONNX)
- Работает на CPU
- Поддерживает русский + английский
- Размер вектора: 384 числа
"""

import gc
import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


# ─────────────────────────────────────────────
# Типы данных
# ─────────────────────────────────────────────


@dataclass
class MatchResult:
    """Результат матчинга — вакансия с оценкой совпадения"""

    vacancy_id: str
    title: str
    company: str
    url: str
    salary_text: str
    score: float
    score_percent: int
    source: str = "hh"

    def format_message(self, rank: int) -> str:
        bar = self._score_bar()
        source_badge = (
            "🌐 RemoteOK"
            if getattr(self, "source", "hh") == "remoteok"
            else "🔵 hh.ru"
        )
        return (
            f"{rank}. {self.title}\n"
            f"🏢 {self.company}\n"
            f"🎯 Совпадение: {bar} {self.score_percent}%\n"
            f"{self.salary_text}"
            f"{source_badge} | 🔗 {self.url}"
        )

    def _score_bar(self) -> str:
        filled = round(self.score_percent / 20)
        return "🟩" * filled + "⬜" * (5 - filled)


# ─────────────────────────────────────────────
# ONNX-обёртка (заменяет PyTorch целиком)
# ─────────────────────────────────────────────


class ONNXEmbedder:
    """
    Загружает модель через ONNX Runtime вместо PyTorch.
    Экономит ~200MB RAM за счёт отсутствия PyTorch.
    Интерфейс совместим с SentenceTransformer.encode().
    """

    def __init__(self, model_name: str):
        logger.info("Загрузка ONNX модели %s...", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_name, export=True
        )
        gc.collect()
        logger.info("ONNX модель загружена ✅")

    def encode(
        self,
        texts,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        """Совместимый с SentenceTransformer.encode() интерфейс"""
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="np",
            )
            outputs = self.model(**inputs)

            # Mean pooling
            token_embeddings = outputs.last_hidden_state
            attention_mask = inputs["attention_mask"]
            mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
            sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
            sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
            embeddings = sum_embeddings / sum_mask

            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.clip(norms, a_min=1e-9, a_max=None)

            all_embeddings.append(embeddings)

        result = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
        gc.collect()

        if single_input and result.ndim == 2 and result.shape[0] == 1:
            return result[0]
        return result


# ─────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────


class EmbeddingPipeline:
    """
    Загружает модель, создаёт векторы, считает совпадение.
    Использует ONNX Runtime для экономии RAM.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = ONNXEmbedder(model_name)

    # ── Создание векторов ──────────────────────

    def embed_resume(self, resume) -> np.ndarray:
        text = self._resume_to_text(resume)
        return self._embed(text)

    def embed_vacancy(self, vacancy: dict) -> np.ndarray:
        text = self._vacancy_to_text(vacancy)
        return self._embed(text)

    def embed_text(self, text: str) -> np.ndarray:
        return self._embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    # ── Матчинг ───────────────────────────────

    def match(
        self,
        resume_vector: np.ndarray,
        vacancies: list[dict],
        top_k: int = 10,
    ) -> list[MatchResult]:
        if not vacancies:
            return []

        vacancy_texts = [self._vacancy_to_text(v) for v in vacancies]
        vacancy_vectors = self.embed_batch(vacancy_texts)
        scores = self._cosine_similarity_batch(resume_vector, vacancy_vectors)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            vacancy = vacancies[idx]
            score = float(scores[idx])
            vac_id = str(vacancy.get("id", ""))
            source = "remoteok" if vac_id.startswith("remoteok_") else "hh"

            if source == "remoteok":
                salary_raw = vacancy.get("_salary_text", "")
                salary_text = f"💰 {salary_raw}\n" if salary_raw else ""
            else:
                salary_text = self._format_salary(vacancy)

            results.append(
                MatchResult(
                    vacancy_id=vac_id,
                    title=vacancy.get("name", ""),
                    company=vacancy.get("employer", {}).get("name", ""),
                    url=vacancy.get("alternate_url", ""),
                    salary_text=salary_text,
                    score=round(score, 3),
                    score_percent=round(score * 100),
                    source=source,
                )
            )

        return results

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(
            np.dot(vec1, vec2)
            / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        )

    # ── Сериализация ──────────────────────────

    @staticmethod
    def vector_to_bytes(vector: np.ndarray) -> bytes:
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_vector(data: bytes) -> np.ndarray:
        return np.frombuffer(data, dtype=np.float32)

    @staticmethod
    def vector_to_list(vector: np.ndarray) -> list[float]:
        return vector.tolist()

    # ── Приватные методы ──────────────────────

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode(
            text,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _cosine_similarity_batch(
        self, query_vector: np.ndarray, matrix: np.ndarray
    ) -> np.ndarray:
        return np.dot(matrix, query_vector)

    def _resume_to_text(self, resume) -> str:
        parts = []
        if resume.desired_position:
            parts.append(resume.desired_position)
            parts.append(resume.desired_position)
        if resume.skills:
            parts.append("Навыки: " + ", ".join(resume.skills))
        if resume.experience_text:
            parts.append(resume.experience_text[:1000])
        if resume.education:
            parts.append(resume.education)
        return "\n".join(parts) if parts else resume.raw_text[:2000]

    def _vacancy_to_text(self, vacancy: dict) -> str:
        parts = []
        title = vacancy.get("name", "")
        if title:
            parts.append(title)
            parts.append(title)
        employer = vacancy.get("employer", {}).get("name", "")
        if employer:
            parts.append(employer)
        snippet = vacancy.get("snippet", {})
        if snippet:
            requirement = snippet.get("requirement", "") or ""
            responsibility = snippet.get("responsibility", "") or ""
            parts.append(requirement)
            parts.append(responsibility)
        description = vacancy.get("description", "") or ""
        if description:
            clean = re.sub(r"<[^>]+>", " ", description)
            parts.append(clean[:800])
        return "\n".join(p for p in parts if p)

    def _format_salary(self, vacancy: dict) -> str:
        salary = vacancy.get("salary")
        if not salary:
            return ""
        from_val = salary.get("from")
        to_val = salary.get("to")
        currency = salary.get("currency", "RUB")
        if from_val and to_val:
            return f"💰 {from_val:,}–{to_val:,} {currency}\n"
        elif from_val:
            return f"💰 от {from_val:,} {currency}\n"
        elif to_val:
            return f"💰 до {to_val:,} {currency}\n"
        return ""


# ─────────────────────────────────────────────
# Синглтон
# ─────────────────────────────────────────────

_pipeline_instance: Optional[EmbeddingPipeline] = None


def get_pipeline() -> EmbeddingPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EmbeddingPipeline()
    return _pipeline_instance