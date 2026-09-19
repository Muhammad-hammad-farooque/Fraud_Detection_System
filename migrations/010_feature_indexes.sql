-- T-18: indexes for the T-16 feature queries, for PostgreSQL.
--
-- The device, merchant and population aggregates filter on these columns.
-- Without the indexes each is a full table scan, which retraining repeats once
-- per labelled row. CONCURRENTLY avoids locking writes on a live table, and
-- cannot run inside a transaction block - so this file has none.

CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_txn_device_created   ON transactions (device_id, created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_txn_merchant_created ON transactions (merchant_id, created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_txn_amount_created   ON transactions (amount, created_at);
