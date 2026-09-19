-- T-13: model registry, for PostgreSQL.
--
-- Stamps each decision with the model version that scored it. Transactions
-- scored before T-13 were all scored by the unversioned baseline.

BEGIN;

ALTER TABLE transactions ADD COLUMN IF NOT EXISTS model_version VARCHAR;
UPDATE transactions SET model_version = 'rf-baseline-unversioned' WHERE model_version IS NULL;

COMMIT;
