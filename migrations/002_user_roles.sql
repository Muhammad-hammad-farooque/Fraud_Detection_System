-- T-10: role-based access control, for PostgreSQL.
-- Every existing account becomes a CUSTOMER. Promote the first admin with:
--     python -m scripts.set_role someone@bank.com ADMIN

BEGIN;

ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR NOT NULL DEFAULT 'CUSTOMER';

COMMIT;
