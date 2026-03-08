"""
embedding_pipeline.py — векторизация резюме и вакансий + матчинг

ИЗМЕНЕНИЯ (OOM-fix):
1. Модель заменена: L12 (118M параметров, ~500MB RAM)
                  → L6  (22M параметров, ~100MB RAM)
   Качество для RU+EN остаётся хорошим, RAM падает в ~5 раз.
2. torch.no_grad() на всех encode — убирает аллокации градиентов.
3. gc.collect() после батч-операций — возвращает RAM ОС.
4. import re вынесен на верхний уровень (не внутри метода).

Модель: paraphrase-multilingual-MiniLM-L6-v2
- Работает на CPU (Railway бесплатный план)
- Поддерживает 50+ языков включая русский и английский
- Размер вектора: 384 числа (тот же что у L12!)
- Размер модели: ~80MB (vs ~470MB у L12)
"""

import gc
import logging
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ─── ГЛАВНОЕ ИЗМЕНЕНИЕ ───────────────────────
# L12 → L6: та же архитектура, те же 384-мерные векторы,
# но 6 слоёв вместо 12 — помещается в 512MB Railway.
MODEL_NAME = "paraphrase-multilingual-MiniLM-L6-v2"


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
    score: float  # 0.0 – 1.0, где 1.0 = идеальное совпадение
    score_percent: int  # score * 100, для отображения
    source: str = "hh"  # "hh" или "remoteok"

    def format_message(self, rank: int) -> str:
        """Форматирует вакансию для отправки в Telegram"""
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
        filled = round(self.score_percent / 20)  # 5 сегментов
        return "🟩" * filled + "⬜" * (5 - filled)


# ─────────────────────────────────────────────
# Основной класс
# ─────────────────────────────────────────────


class EmbeddingPipeline:
    """
    Загружает модель, создаёт векторы, считает совпадение.

    Использование:
        pipeline = EmbeddingPipeline()
        resume_vec = pipeline.embed_resume(resume)
        matches = pipeline.match(resume_vec, vacancies, top_k=10)
    """

    def __init__(self, model_name: str = MODEL_NAME):
        logger.info("Загрузка модели %s...", model_name)
        self.model = SentenceTransformer(model_name, device="cpu")
        logger.info("Модель загружена ✅")

    # ── Создание векторов ──────────────────────

    def embed_resume(self, resume) -> np.ndarray:
        """
        Принимает ParsedResume, возвращает вектор 384 чисел.
        Собирает текст из всех полей — чем больше данных, тем точнее вектор.
        """
        text = self._resume_to_text(resume)
        return self._embed(text)

    def embed_vacancy(self, vacancy: dict) -> np.ndarray:
        """Принимает словарь вакансии (hh.ru формат), возвращает вектор."""
        text = self._vacancy_to_text(vacancy)
        return self._embed(text)

    def embed_text(self, text: str) -> np.ndarray:
        """Векторизует произвольный текст"""
        return self._embed(text)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Векторизует список текстов разом — быстрее чем по одному.
        Возвращает матрицу shape (len(texts), 384)
        """
        if not texts:
            return np.array([])
        with torch.no_grad():
            result = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        gc.collect()
        return result

    # ── Матчинг ───────────────────────────────

    def match(
        self,
        resume_vector: np.ndarray,
        vacancies: list[dict],
        top_k: int = 10,
    ) -> list[MatchResult]:
        """
        Главный метод матчинга.

        resume_vector — вектор резюме из embed_resume()
        vacancies — список вакансий в формате hh.ru API
        top_k — сколько лучших вернуть

        Возвращает список MatchResult, отсортированный по убыванию score.
        """
        if not vacancies:
            return []

        # Векторизуем все вакансии разом (батч — быстро)
        vacancy_texts = [self._vacancy_to_text(v) for v in vacancies]
        vacancy_vectors = self.embed_batch(vacancy_texts)

        # Считаем cosine similarity со всеми вакансиями
        scores = self._cosine_similarity_batch(resume_vector, vacancy_vectors)

        # Берём топ-K индексов
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            vacancy = vacancies[idx]
            score = float(scores[idx])

            # Определяем источник
            vac_id = str(vacancy.get("id", ""))
            source = "remoteok" if vac_id.startswith("remoteok_") else "hh"

            # Для Remotive берём salary_text из _salary_text поля
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
        """Cosine similarity между двумя векторами. Результат: 0.0 – 1.0"""
        return float(
            np.dot(vec1, vec2)
            / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-10)
        )

    # ── Сериализация векторов ─────────────────

    @staticmethod
    def vector_to_bytes(vector: np.ndarray) -> bytes:
        """Конвертирует вектор в bytes для хранения в PostgreSQL (bytea)"""
        return vector.astype(np.float32).tobytes()

    @staticmethod
    def bytes_to_vector(data: bytes) -> np.ndarray:
        """Восстанавливает вектор из bytes"""
        return np.frombuffer(data, dtype=np.float32)

    @staticmethod
    def vector_to_list(vector: np.ndarray) -> list[float]:
        """Конвертирует в список для pgvector"""
        return vector.tolist()

    # ── Приватные методы ──────────────────────

    def _embed(self, text: str) -> np.ndarray:
        """Создаёт нормализованный вектор для одного текста"""
        with torch.no_grad():
            result = self.model.encode(
                text,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        return result

    def _cosine_similarity_batch(
        self,
        query_vector: np.ndarray,
        matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Быстрое вычисление cosine similarity между одним вектором и матрицей.
        Поскольку normalize_embeddings=True, можно просто dot product.
        """
        return np.dot(matrix, query_vector)

    def _resume_to_text(self, resume) -> str:
        """
        Собирает текст резюме для векторизации.
        Порядок важен — желаемая должность и навыки идут первыми,
        они важнее для матчинга.
        """
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
        """Собирает текст вакансии для векторизации"""
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
        """Форматирует зарплату для вывода"""
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
# Синглтон — одна модель на весь процесс бота
# ─────────────────────────────────────────────

_pipeline_instance: Optional[EmbeddingPipeline] = None


def get_pipeline() -> EmbeddingPipeline:
    """
    Возвращает единственный экземпляр EmbeddingPipeline.
    Модель загружается один раз при первом вызове.
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = EmbeddingPipeline()
    return _pipeline_instance