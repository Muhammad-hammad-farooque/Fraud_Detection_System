-- T-16: sign-up time on users, for the account_age_days feature. PostgreSQL.
--
-- No sign-up time was ever recorded. The best available approximation for an
-- existing account is its first transaction; accounts with none get NOW().
-- New accounts are stamped by the application at registration.

BEGIN;

ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;

UPDATE users u
SET created_at = COALESCE(
    (SELECT MIN(t.created_at) FROM transactions t WHERE t.user_id = u.id),
    NOW()
)
WHERE u.created_at IS NULL;

COMMIT;
