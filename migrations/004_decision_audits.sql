-- T-11: immutable decision audit trail, for PostgreSQL.
--
-- The ORM already refuses to update or delete these rows. The trigger enforces
-- the same rule for anything that bypasses the application: a psql session, a
-- maintenance script, a future service. Only a superuser dropping the trigger
-- can change history, and that is itself visible.

BEGIN;

CREATE TABLE IF NOT EXISTS decision_audits (
    id             SERIAL PRIMARY KEY,
    transaction_id INTEGER NOT NULL REFERENCES transactions (id),
    feature_vector JSONB NOT NULL,
    rule_hits      JSONB NOT NULL,
    rule_score     DOUBLE PRECISION NOT NULL,
    model_prob     DOUBLE PRECISION NOT NULL,
    final_score    DOUBLE PRECISION NOT NULL,
    decision       VARCHAR NOT NULL,
    model_version  VARCHAR NOT NULL,
    policy_version VARCHAR NOT NULL,
    scoring_params JSONB NOT NULL,
    policy_config  JSONB NOT NULL,
    policy_context JSONB NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_decision_audits_transaction_id ON decision_audits (transaction_id);

CREATE OR REPLACE FUNCTION refuse_decision_audit_change() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'decision_audits is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS decision_audits_append_only ON decision_audits;
CREATE TRIGGER decision_audits_append_only
    BEFORE UPDATE OR DELETE ON decision_audits
    FOR EACH ROW EXECUTE FUNCTION refuse_decision_audit_change();

-- Transactions scored before this migration have no audit row, and cannot be
-- given a faithful one: their feature vectors were never stored. They stay
-- unaudited rather than receiving a reconstructed record that would look
-- authoritative and is not.

COMMIT;
