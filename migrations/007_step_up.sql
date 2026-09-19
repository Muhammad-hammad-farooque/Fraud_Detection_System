-- T-14b: step-up authentication challenges, for PostgreSQL.

BEGIN;

CREATE TABLE IF NOT EXISTS step_up_challenges (
    id             SERIAL PRIMARY KEY,
    transaction_id INTEGER NOT NULL UNIQUE REFERENCES transactions (id),
    method         VARCHAR NOT NULL,
    code_hash      VARCHAR(64) NOT NULL,
    status         VARCHAR NOT NULL DEFAULT 'PENDING',
    attempts       INTEGER NOT NULL DEFAULT 0,
    max_attempts   INTEGER NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL,
    expires_at     TIMESTAMPTZ NOT NULL,
    resolved_at    TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_step_up_status_expires ON step_up_challenges (status, expires_at);

-- Transactions already sitting in STEP_UP were never challenged, and the
-- customer has long since left the checkout. Settle them the way an expired
-- challenge would be: rejected, beside the engine's untouched decision.
UPDATE transactions
SET resolved_decision = 'REJECT'
WHERE decision = 'STEP_UP' AND resolved_decision IS NULL;

COMMIT;
