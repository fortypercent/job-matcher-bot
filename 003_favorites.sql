CREATE TABLE IF NOT EXISTS favorites (
    id          SERIAL PRIMARY KEY,
    user_id     BIGINT REFERENCES users(id) ON DELETE CASCADE,
    vacancy_id  TEXT NOT NULL,
    title       TEXT,
    company     TEXT,
    url         TEXT,
    salary_text TEXT,
    saved_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, vacancy_id)
);
