from .database import Sessionlocal

def get_db():
    db = Sessionlocal()
    try:
        yield db
    finally:
        db.close()