"""Step-up authentication: challenge, verify, expire (T-14b).

A STEP_UP decision is friction instead of a decline: the customer proves it is
them, and the payment goes through. This module owns that lifecycle. Every
function that changes state works inside the caller's database transaction and
never commits on its own; delivery happens only after the caller has committed,
so a rolled-back payment never sends anyone a code.
"""
import hashlib
import hmac
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from sqlalchemy.orm import Session

from .. import models
from ..auth import SECRET_KEY
from ..features import as_utc
from ..models import CaseSource, ChallengeStatus
from .case_service import open_case

logger = logging.getLogger(__name__)

OTP_METHOD = "OTP"
OTP_DIGITS = 6


@dataclass(frozen=True)
class StepUpConfig:
    ttl_seconds: int
    max_attempts: int
    dev_echo: bool          # put the code in the API response - development only


def load_step_up_config() -> StepUpConfig:
    return StepUpConfig(
        ttl_seconds=int(os.getenv("STEP_UP_TTL_SECONDS", "300")),
        max_attempts=int(os.getenv("STEP_UP_MAX_ATTEMPTS", "3")),
        dev_echo=os.getenv("STEP_UP_DEV_ECHO", "false").lower() == "true",
    )


class ChallengeSender(Protocol):
    """Delivers a code to the customer out of band: SMS, email, push, 3-D Secure."""

    def send(self, user: models.User, challenge: models.StepUpChallenge, code: str) -> None: ...


class LogSender:
    """Development stand-in for a real delivery channel. Never use in production:
    it writes the code to the server log."""

    def send(self, user: models.User, challenge: models.StepUpChallenge, code: str) -> None:
        logger.warning("DEV step-up code for %s, challenge %s: %s", user.email, challenge.id, code)


_sender: ChallengeSender = LogSender()


def get_sender() -> ChallengeSender:
    return _sender


def set_sender(sender: ChallengeSender) -> ChallengeSender:
    """Install a delivery channel. Returns the previous one, so tests can restore it."""
    global _sender
    previous, _sender = _sender, sender
    return previous


class StepUpError(Exception):
    """A lifecycle rule was broken. `status_code` is the HTTP status it maps to."""

    def __init__(self, message: str, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


def _hash_code(challenge_id: int, code: str) -> str:
    # Keyed with the server secret, so a leaked table cannot be brute-forced
    # offline across the million possible six-digit codes.
    message = f"{challenge_id}:{code}".encode("utf-8")
    return hmac.new(SECRET_KEY.encode("utf-8"), message, hashlib.sha256).hexdigest()


def _new_code() -> str:
    return f"{secrets.randbelow(10 ** OTP_DIGITS):0{OTP_DIGITS}d}"


def issue_challenge(
    db: Session,
    transaction: models.Transaction,
    now: datetime | None = None,
    cfg: StepUpConfig | None = None,
) -> tuple[models.StepUpChallenge, str]:
    """Create a PENDING challenge for a STEP_UP transaction. Returns it with the
    plaintext code, which the caller delivers after committing and then drops."""
    cfg = cfg or load_step_up_config()
    now = now or datetime.now(timezone.utc)
    challenge = models.StepUpChallenge(
        transaction_id=transaction.id,
        method=OTP_METHOD,
        code_hash="pending",
        status=ChallengeStatus.PENDING.value,
        attempts=0,
        max_attempts=cfg.max_attempts,
        created_at=now,
        expires_at=now + timedelta(seconds=cfg.ttl_seconds),
    )
    db.add(challenge)
    db.flush()                                   # the id is part of the hashed message
    code = _new_code()
    challenge.code_hash = _hash_code(challenge.id, code)
    return challenge, code


def _settle(challenge: models.StepUpChallenge, status: ChallengeStatus, now: datetime,
            resolved_decision: str | None) -> None:
    challenge.status = status.value
    challenge.resolved_at = now
    if resolved_decision is not None:
        # The engine's own `decision` stays STEP_UP forever; the outcome of the
        # challenge goes beside it, the same rule T-12 follows.
        challenge.transaction.resolved_decision = resolved_decision


def _expire_if_due(challenge: models.StepUpChallenge, now: datetime) -> bool:
    if challenge.status == ChallengeStatus.PENDING.value and now > as_utc(challenge.expires_at):
        _settle(challenge, ChallengeStatus.EXPIRED, now, "REJECT")
        return True
    return False


def verify(
    db: Session,
    challenge: models.StepUpChallenge,
    code: str,
    now: datetime | None = None,
) -> models.StepUpChallenge:
    """Check a submitted code and move the challenge on.

    Correct code: PASSED, and the payment is allowed. Wrong code: an attempt is
    spent; spending the last one marks the challenge FAILED and opens a case,
    leaving the payment held for an analyst rather than silently rejected -
    repeated failures are an account-takeover signal worth a human look. Past
    its deadline the challenge EXPIRES and the payment is rejected.
    """
    now = now or datetime.now(timezone.utc)
    if challenge.status != ChallengeStatus.PENDING.value:
        raise StepUpError(f"Challenge is already {challenge.status}")
    if _expire_if_due(challenge, now):
        return challenge

    if hmac.compare_digest(challenge.code_hash, _hash_code(challenge.id, code)):
        _settle(challenge, ChallengeStatus.PASSED, now, "ALLOW")
        return challenge

    challenge.attempts += 1
    if challenge.attempts >= challenge.max_attempts:
        _settle(challenge, ChallengeStatus.FAILED, now, resolved_decision=None)
        open_case(db, challenge.transaction, CaseSource.STEP_UP_FAILED, now=now)
    return challenge


def expire_stale(db: Session, now: datetime | None = None) -> int:
    """Settle every pending challenge past its deadline. Returns how many.

    Run on a schedule (scripts/expire_challenges.py) so abandoned challenges
    are resolved even if the customer never comes back.
    """
    now = now or datetime.now(timezone.utc)
    pending = (
        db.query(models.StepUpChallenge)
        .filter(models.StepUpChallenge.status == ChallengeStatus.PENDING.value)
        .all()
    )
    return sum(1 for challenge in pending if _expire_if_due(challenge, now))
