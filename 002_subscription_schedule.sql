ALTER TABLE subscriptions
    ADD COLUMN IF NOT EXISTS frequency   TEXT DEFAULT 'daily'
        CHECK (frequency IN ('daily', 'twice_daily', 'weekly', 'monthly')),
    ADD COLUMN IF NOT EXISTS days        INTEGER[] DEFAULT '{1,2,3,4,5}',
    ADD COLUMN IF NOT EXISTS send_hour   INTEGER DEFAULT 9,
    ADD COLUMN IF NOT EXISTS send_minute INTEGER DEFAULT 0;
