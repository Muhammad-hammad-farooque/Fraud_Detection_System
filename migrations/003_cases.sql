-- T-12: analyst case queue, for PostgreSQL.

BEGIN;

-- The human outcome of a review, beside the engine's untouched decision.
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS resolved_decision VARCHAR;

CREATE TABLE IF NOT EXISTS cases (
    id             SERIAL PRIMARY KEY,
    transaction_id INTEGER NOT NULL UNIQUE REFERENCES transactions (id),
    claim_id       INTEGER REFERENCES claims (id),
    source         VARCHAR NOT NULL,
    status         VARCHAR NOT NULL DEFAULT 'OPEN',
    priority       VARCHAR NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL,
    sla_due_at     TIMESTAMPTZ NOT NULL,
    assigned_to    INTEGER REFERENCES users (id),
    assigned_at    TIMESTAMPTZ,
    resolved_by    INTEGER REFERENCES users (id),
    resolved_at    TIMESTAMPTZ,
    notes          VARCHAR
);

CREATE INDEX IF NOT EXISTS ix_case_status_created ON cases (status, created_at);

-- Transactions already sitting in REVIEW before this migration have no case and
-- would stay dead ends. Open one for each, due now, so they surface first.
INSERT INTO cases (transaction_id, source, status, priority, created_at, sla_due_at)
SELECT t.id, 'POLICY_REVIEW', 'OPEN', 'NORMAL', NOW(), NOW()
FROM transactions t
LEFT JOIN cases c ON c.transaction_id = t.id
WHERE t.decision IN ('REVIEW', 'MANUAL_CHECK') AND c.id IS NULL;

COMMIT;
