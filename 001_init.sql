-- Удаляем если есть (для чистого перезапуска)
DROP TABLE IF EXISTS vacancies CASCADE;
DROP TABLE IF EXISTS subscriptions CASCADE;
DROP TABLE IF EXISTS resumes CASCADE;

-- ── Резюме ────────────────────────────────────
CREATE TABLE IF NOT EXISTS resumes (
    id              SERIAL PRIMARY KEY,
    user_id         BIGINT REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT,
    position        TEXT,
    skills          TEXT[],
    experience_years FLOAT,
    education       TEXT,
    raw_text        TEXT,
    embedding       TEXT,   -- JSON строка: '[0.1, 0.2, ...]'
    source          TEXT DEFAULT 'pdf' CHECK (source IN ('pdf', 'hh')),
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Подписки ──────────────────────────────────
CREATE TABLE IF NOT EXISTS subscriptions (
    id              SERIAL PRIMARY KEY,
    user_id         BIGINT REFERENCES users(id) ON DELETE CASCADE,
    resume_id       INT REFERENCES resumes(id) ON DELETE CASCADE,
    search_query    TEXT NOT NULL,
    area            INT DEFAULT 1,
    active          BOOLEAN DEFAULT TRUE,
    last_sent_at    TIMESTAMPTZ,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── Вакансии (кэш) ────────────────────────────
CREATE TABLE IF NOT EXISTS vacancies (
    id              TEXT PRIMARY KEY,
    title           TEXT,
    company         TEXT,
    url             TEXT,
    salary_from     INT,
    salary_to       INT,
    currency        TEXT,
    snippet         TEXT,
    embedding       TEXT,   -- JSON строка
    source          TEXT DEFAULT 'hh',
    published_at    TIMESTAMPTZ,
    cached_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS vacancies_published_idx
    ON vacancies (published_at DESC);