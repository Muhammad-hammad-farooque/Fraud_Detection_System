from sqlalchemy import Column, Integer, Float, ForeignKey, String, DateTime
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String,)
    email = Column(String, unique=True)

class Transaction(Base):
    __tablename__ = "transaction"

    id = Column(Integer, primary_key=True, index=True)
    location = Column(String)
    amount = Column(Float)
    device_id = Column(String)
    user_id = Column(Integer, ForeignKey("Users.id"))
    created_at= Column(DateTime, defualt = datetime.utcnow())

    user = relationship("User", back_populates="transaction")