from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set. Check your .env file.")

# Pin PostgreSQL sessions to UTC: typical_hour_share extracts the hour from a
# timestamptz, which PostgreSQL would otherwise report in the server's zone.
_connect_args = {"options": "-c timezone=utc"} if DATABASE_URL.startswith("postgresql") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)

SessionLocal = sessionmaker(autoflush=False, autocommit=False, bind=engine)

Base= declarative_base()