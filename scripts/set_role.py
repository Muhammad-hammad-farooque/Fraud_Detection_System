"""
Set a user's role from the command line.

The admin endpoint needs an admin to call it, so the first admin has to be
created out of band. Run this once after registering the account:

Usage:
    python -m scripts.set_role someone@bank.com ADMIN
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal
from app import models


def set_role(db, email: str, role: str) -> models.User:
    """Assign `role` to the user with `email`. Raises ValueError if either is unknown."""
    try:
        new_role = models.Role(role.upper())
    except ValueError:
        valid = ", ".join(r.value for r in models.Role)
        raise ValueError(f"Unknown role {role!r}; expected one of: {valid}")

    user = db.query(models.User).filter(models.User.email == email).first()
    if user is None:
        raise ValueError(f"No user registered with email {email!r}")

    user.role = new_role.value
    db.commit()
    return user


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)

    db = SessionLocal()
    try:
        user = set_role(db, sys.argv[1], sys.argv[2])
        print(f"{user.email} is now {user.role}")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    finally:
        db.close()
