CREATE TABLE IF NOT EXISTS user_preferences (
    user_id     BIGINT PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    areas       INTEGER[] DEFAULT '{1}',   -- hh.ru area IDs
    area_names  TEXT[]    DEFAULT '{"Москва"}',
    salary_from INTEGER,
    salary_to   INTEGER,
    remote_only BOOLEAN DEFAULT FALSE,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
