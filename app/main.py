from fastapi import FastAPI
from .config import get_rules_config
from .database import engine, Base
from .routers import user_route, transactions, claims, auth, analyst, admin

# Fail fast: a malformed rules file stops the API here, at startup, rather than
# at the first payment. The active model is checked the same way when
# app.ML.models is imported by the routers below.
get_rules_config()

app = FastAPI(title="Fraud Detection System", version="1.0.0")

Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(user_route.router)
app.include_router(transactions.router)
app.include_router(claims.router)
app.include_router(analyst.router)
app.include_router(admin.router)


@app.get("/")
def root():
    return {"message": "Fraud Detection System is running"}
