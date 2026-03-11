from fastapi import FastAPI
from .database import engine, Base
from .routers import user_route, transactions

app = FastAPI(title="Fraud Detection System", version="1.0.0")

Base.metadata.create_all(bind=engine)

app.include_router(user_route.router)
app.include_router(transactions.router)

@app.get("/")
def root():
    return {"message":"The Server is Running"}