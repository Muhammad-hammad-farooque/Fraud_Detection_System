-- T-15: payment context on transactions, for PostgreSQL.
--
-- All columns are nullable: existing rows and clients that do not send these
-- fields stay valid. Nothing is backfilled - there is no source for the values.

BEGIN;

ALTER TABLE transactions ADD COLUMN IF NOT EXISTS merchant_id       VARCHAR(64);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS merchant_category VARCHAR(4);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS currency          VARCHAR(3);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS channel           VARCHAR(8);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS ip_address        VARCHAR(45);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS card_token        VARCHAR(64);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS external_txn_id   VARCHAR(128);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS latitude          DOUBLE PRECISION;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS longitude         DOUBLE PRECISION;

COMMIT;
