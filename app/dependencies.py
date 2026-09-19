from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from .database import SessionLocal
from .auth import decode_access_token
from . import models

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> models.User:
    """Decode JWT token and return the logged-in user."""
    user_id = decode_access_token(token)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


def require_role(*allowed: models.Role):
    """Dependency factory. Raises 403 when current_user.role is not in allowed.

    The role is read from the database on every request rather than from the
    token, so a demotion takes effect immediately instead of when the token
    expires.
    """
    allowed_values = {role.value for role in allowed}

    def dependency(current_user: models.User = Depends(get_current_user)) -> models.User:
        if current_user.role not in allowed_values:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient role for this endpoint",
            )
        return current_user

    return dependency


# Customer-scoped routes. Staff accounts do not make payments or file disputes:
# they would otherwise gain a second, silent path into customer data flows, and
# their own activity would be mixed into the labels analysts produce.
require_customer = require_role(models.Role.CUSTOMER)

# Analyst routes see every customer's records. Admins inherit analyst access.
require_analyst = require_role(models.Role.ANALYST, models.Role.ADMIN)

require_admin = require_role(models.Role.ADMIN)
