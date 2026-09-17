-- Phase A schema changes (T-02, T-04, T-05, T-08), for PostgreSQL.
--
-- The application still calls Base.metadata.create_all() on boot, which creates
-- new tables but cannot alter existing columns. Run this once against a database
-- created before Phase A. T-21 replaces this file with Alembic and baselines the
-- schema properly.

BEGIN;

-- T-02: every scoring query filters on this pair.
CREATE INDEX IF NOT EXISTS ix_txn_user_created ON transactions (user_id, created_at);

-- T-04: record which policy produced each decision.
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS policy_version VARCHAR;

-- T-05: a prediction is not a label.
ALTER TABLE transactions RENAME COLUMN is_fraud TO predicted_fraud;

-- T-05: confirmed ground truth, one row per transaction.
CREATE TABLE IF NOT EXISTS transaction_outcomes (
    id                 SERIAL PRIMARY KEY,
    transaction_id     INTEGER NOT NULL UNIQUE REFERENCES transactions (id),
    is_fraud_confirmed BOOLEAN NOT NULL,
    source             VARCHAR NOT NULL,
    confirmed_by       INTEGER REFERENCES users (id),
    confirmed_at       TIMESTAMPTZ NOT NULL,
    notes              VARCHAR
);

-- T-08: timezone-aware timestamps, so no caller has to patch a naive value.
ALTER TABLE transactions        ALTER COLUMN created_at   TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC';
ALTER TABLE claims              ALTER COLUMN created_at   TYPE TIMESTAMPTZ USING created_at AT TIME ZONE 'UTC';
ALTER TABLE transaction_outcomes ALTER COLUMN confirmed_at TYPE TIMESTAMPTZ USING confirmed_at AT TIME ZONE 'UTC';

COMMIT;
