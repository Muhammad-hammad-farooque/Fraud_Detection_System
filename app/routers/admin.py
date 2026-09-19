from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from .. import models, schemas
from ..dependencies import get_db, require_admin

router = APIRouter(
    prefix="/admin",
    tags=["Admin"]
)


@router.get("/users", response_model=List[schemas.UserResponse])
def list_users(
    db: Session = Depends(get_db),
    admin: models.User = Depends(require_admin)
):
    """List every account with its role."""
    return db.query(models.User).order_by(models.User.id).all()


@router.patch("/users/{user_id}/role", response_model=schemas.UserResponse)
def set_user_role(
    user_id: int,
    update: schemas.RoleUpdate,
    db: Session = Depends(get_db),
    admin: models.User = Depends(require_admin)
):
    """Change a user's role. Admins cannot demote themselves, so the system can
    never be left without an admin by accident."""
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if user.id == admin.id and update.role != models.Role.ADMIN:
        raise HTTPException(status_code=400, detail="Admins cannot demote themselves")

    user.role = update.role.value
    db.commit()
    db.refresh(user)
    return user
