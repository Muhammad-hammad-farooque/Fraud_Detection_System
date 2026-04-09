from fastapi import FastAPI
from .database import engine, Base
from .routers import user_route, transactions, claims, auth

app = FastAPI(title="Fraud Detection System", version="1.0.0")

Base.metadata.create_all(bind=engine)

app.include_router(auth.router)
app.include_router(user_route.router)
app.include_router(transactions.router)
app.include_router(claims.router)


@app.get("/")
def root():
    return {"message": "Fraud Detection System is running"}
