-- T-14: idempotency keys on transactions, for PostgreSQL.
--
-- Existing rows get NULL keys. PostgreSQL treats NULLs as distinct under a
-- unique constraint, so requests sent without the header are unaffected.

BEGIN;

ALTER TABLE transactions ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(255);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS idempotency_fingerprint VARCHAR(64);

ALTER TABLE transactions
    ADD CONSTRAINT uq_txn_user_idempotency_key UNIQUE (user_id, idempotency_key);

COMMIT;
