"""
database.py — работа с PostgreSQL

Зависимости:
    pip install asyncpg==0.29.0

Переменная окружения:
    DATABASE_URL=postgresql://user:pass@host:5432/dbname
"""

import os
import logging
import asyncpg
from typing import Optional

logger = logging.getLogger(__name__)

_pool: Optional[asyncpg.Pool] = None


# ─────────────────────────────────────────────
# Подключение
# ─────────────────────────────────────────────

async def get_pool() -> asyncpg.Pool:
    """Возвращает пул соединений. Создаёт при первом вызове."""
    global _pool
    if _pool is None:
        url = os.getenv("DATABASE_URL")
        if not url:
            raise RuntimeError("DATABASE_URL не задан в переменных окружения")

        _pool = await asyncpg.create_pool(
            url,
            min_size=1,
            max_size=10,
            command_timeout=30,
        )
        logger.info("✅ Подключение к PostgreSQL установлено")

    return _pool


async def close_pool():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def run_migrations():
    """Запускает SQL миграции из папки migrations/"""
    pool = await get_pool()
    migrations_dir = os.path.join(os.path.dirname(__file__), "migrations")

    if not os.path.exists(migrations_dir):
        logger.warning("Папка migrations/ не найдена")
        return

    sql_files = sorted(f for f in os.listdir(migrations_dir) if f.endswith(".sql"))

    async with pool.acquire() as conn:
        for filename in sql_files:
            path = os.path.join(migrations_dir, filename)
            with open(path) as f:
                sql = f.read()
            await conn.execute(sql)
            logger.info(f"✅ Миграция применена: {filename}")


# ─────────────────────────────────────────────
# Пользователи
# ─────────────────────────────────────────────

async def upsert_user(user_id: int, username: str = None, full_name: str = None):
    """Создаёт пользователя или обновляет last_active_at"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO users (id, username, full_name)
            VALUES ($1, $2, $3)
            ON CONFLICT (id) DO UPDATE SET
                username = EXCLUDED.username,
                full_name = EXCLUDED.full_name,
                last_active_at = NOW()
        """, user_id, username, full_name)


async def get_user(user_id: int) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM users WHERE id = $1", user_id)
        return dict(row) if row else None


# ─────────────────────────────────────────────
# Резюме
# ─────────────────────────────────────────────

async def save_resume(user_id: int, resume, embedding: list[float]) -> int:
    """
    Сохраняет резюме в БД. Если у пользователя уже есть резюме — обновляет.
    Возвращает resume_id.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        # Проверяем есть ли уже резюме
        existing = await conn.fetchrow(
            "SELECT id FROM resumes WHERE user_id = $1 ORDER BY created_at DESC LIMIT 1",
            user_id
        )

        if existing:
            # Обновляем существующее
            row = await conn.fetchrow("""
                UPDATE resumes SET
                    name = $2,
                    position = $3,
                    skills = $4,
                    experience_years = $5,
                    education = $6,
                    raw_text = $7,
                    embedding = $8,
                    source = $9,
                    updated_at = NOW()
                WHERE id = $1
                RETURNING id
            """,
                existing['id'],
                resume.name,
                resume.desired_position,
                resume.skills,
                resume.experience_years,
                resume.education,
                resume.raw_text[:5000],
                str(embedding),  # pgvector принимает строку '[0.1, 0.2, ...]'
                resume.source,
            )
        else:
            # Создаём новое
            row = await conn.fetchrow("""
                INSERT INTO resumes
                    (user_id, name, position, skills, experience_years,
                     education, raw_text, embedding, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                RETURNING id
            """,
                user_id,
                resume.name,
                resume.desired_position,
                resume.skills,
                resume.experience_years,
                resume.education,
                resume.raw_text[:5000],
                str(embedding),
                resume.source,
            )

        return row['id']


async def get_resume(user_id: int) -> Optional[dict]:
    """Возвращает последнее резюме пользователя"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT * FROM resumes
            WHERE user_id = $1
            ORDER BY updated_at DESC
            LIMIT 1
        """, user_id)
        return dict(row) if row else None


async def find_similar_resumes(embedding: list[float], limit: int = 10) -> list[dict]:
    """
    HR матчинг — ищет резюме похожие на вакансию.
    Использует векторный индекс pgvector.
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT
                r.*,
                1 - (r.embedding <=> $1::vector) AS score
            FROM resumes r
            ORDER BY r.embedding <=> $1::vector
            LIMIT $2
        """, str(embedding), limit)
        return [dict(row) for row in rows]


# ─────────────────────────────────────────────
# Подписки
# ─────────────────────────────────────────────

async def create_subscription(
    user_id: int,
    resume_id: int,
    search_query: str,
    area: int = 1,
    frequency: str = "daily",
    days: list[int] = None,
    send_hour: int = 9,
    send_minute: int = 0,
) -> int:
    """Создаёт подписку на рассылку с настройками расписания"""
    pool = await get_pool()
    if days is None:
        days = [1, 2, 3, 4, 5]  # пн-пт по умолчанию

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE subscriptions SET active = FALSE WHERE user_id = $1",
            user_id
        )
        row = await conn.fetchrow("""
            INSERT INTO subscriptions
                (user_id, resume_id, search_query, area,
                 frequency, days, send_hour, send_minute)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
        """, user_id, resume_id, search_query, area,
            frequency, days, send_hour, send_minute)
        return row['id']


async def get_active_subscriptions() -> list[dict]:
    """Возвращает все активные подписки для рассылки"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT s.*, r.embedding, r.position, r.skills
            FROM subscriptions s
            JOIN resumes r ON r.id = s.resume_id
            WHERE s.active = TRUE
        """)
        return [dict(row) for row in rows]


async def update_subscription_sent(subscription_id: int):
    """Обновляет время последней отправки"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE subscriptions SET last_sent_at = NOW() WHERE id = $1",
            subscription_id
        )


async def get_subscription(user_id: int) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT * FROM subscriptions
            WHERE user_id = $1 AND active = TRUE
            ORDER BY created_at DESC LIMIT 1
        """, user_id)
        return dict(row) if row else None


async def deactivate_subscription(user_id: int):
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE subscriptions SET active = FALSE WHERE user_id = $1",
            user_id
        )


# ─────────────────────────────────────────────
# Вакансии (кэш)
# ─────────────────────────────────────────────

async def cache_vacancy(vacancy: dict, embedding: list[float]):
    """Сохраняет вакансию в кэш. Игнорирует дубли."""
    pool = await get_pool()
    salary = vacancy.get('salary') or {}

    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO vacancies
                (id, title, company, url, salary_from, salary_to,
                 currency, snippet, embedding, published_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO NOTHING
        """,
            str(vacancy.get('id', '')),
            vacancy.get('name'),
            vacancy.get('employer', {}).get('name'),
            vacancy.get('alternate_url'),
            salary.get('from'),
            salary.get('to'),
            salary.get('currency'),
            _get_snippet(vacancy),
            str(embedding),
            vacancy.get('published_at'),
        )


async def is_vacancy_seen(vacancy_id: str) -> bool:
    """Проверяет была ли вакансия уже закэширована"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM vacancies WHERE id = $1", vacancy_id
        )
        return row is not None


def _get_snippet(vacancy: dict) -> str:
    snippet = vacancy.get('snippet', {}) or {}
    parts = [
        snippet.get('requirement', '') or '',
        snippet.get('responsibility', '') or '',
    ]
    return ' '.join(p for p in parts if p)[:500]


# ─────────────────────────────────────────────
# Избранное
# ─────────────────────────────────────────────

async def add_favorite(user_id: int, vacancy_id: str, title: str, company: str, url: str, salary_text: str = ""):
    """Добавляет вакансию в избранное. Игнорирует если уже есть."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO favorites (user_id, vacancy_id, title, company, url, salary_text)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (user_id, vacancy_id) DO NOTHING
        """, user_id, vacancy_id, title, company, url, salary_text)


async def remove_favorite(user_id: int, vacancy_id: str):
    """Удаляет вакансию из избранного."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM favorites WHERE user_id = $1 AND vacancy_id = $2",
            user_id, vacancy_id
        )


async def is_favorite(user_id: int, vacancy_id: str) -> bool:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id FROM favorites WHERE user_id = $1 AND vacancy_id = $2",
            user_id, vacancy_id
        )
        return row is not None


async def get_favorites(user_id: int) -> list[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT * FROM favorites
            WHERE user_id = $1
            ORDER BY saved_at DESC
        """, user_id)
        return [dict(row) for row in rows]


# ─────────────────────────────────────────────
# Настройки поиска (фильтры)
# ─────────────────────────────────────────────

async def get_preferences(user_id: int) -> dict:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM user_preferences WHERE user_id = $1", user_id
        )
        if row:
            return dict(row)
        # Дефолтные настройки
        return {
            'user_id': user_id,
            'areas': [1],
            'area_names': ['Москва'],
            'salary_from': None,
            'salary_to': None,
            'remote_only': False,
            'show_without_salary': True,
            'experience': ['noExperience', 'between1And3', 'between3And6', 'moreThan6'],
        }


async def save_preferences(user_id: int, areas: list, area_names: list,
                           salary_from: int = None, salary_to: int = None,
                           remote_only: bool = False,
                           show_without_salary: bool = True,
                           experience: list = None):
    if experience is None:
        experience = ['noExperience', 'between1And3', 'between3And6', 'moreThan6']
    pool = await get_pool()
    async with pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO user_preferences
                (user_id, areas, area_names, salary_from, salary_to, remote_only,
                 show_without_salary, experience)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (user_id) DO UPDATE SET
                areas               = EXCLUDED.areas,
                area_names          = EXCLUDED.area_names,
                salary_from         = EXCLUDED.salary_from,
                salary_to           = EXCLUDED.salary_to,
                remote_only         = EXCLUDED.remote_only,
                show_without_salary = EXCLUDED.show_without_salary,
                experience          = EXCLUDED.experience,
                updated_at          = NOW()
        """, user_id, areas, area_names, salary_from, salary_to, remote_only,
            show_without_salary, experience)


async def get_all_resumes() -> list[dict]:
    """Возвращает все резюме из БД для HR поиска"""
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT r.user_id, r.name, r.position, r.skills, r.experience_years,
                   r.embedding, u.username
            FROM resumes r
            JOIN users u ON u.id = r.user_id
            WHERE r.embedding IS NOT NULL
            ORDER BY r.updated_at DESC
        """)
        return [dict(r) for r in rows]