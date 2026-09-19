"""
Settle step-up challenges that passed their deadline without an answer.

Expiry is also applied lazily when a customer answers a challenge, but a
customer who never comes back would otherwise leave the payment pending
forever. Run this on a schedule, e.g. every minute:

Usage:
    python -m scripts.expire_challenges
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app.services.step_up import expire_stale


def run() -> int:
    db = SessionLocal()
    try:
        expired = expire_stale(db)
        db.commit()
        print(f"Expired {expired} step-up challenge(s)")
        return expired
    finally:
        db.close()


if __name__ == "__main__":
    run()
