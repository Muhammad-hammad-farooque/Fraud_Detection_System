from pydantic import BaseModel

class UserCreate(BaseModel):
    name:str
    email:str

class TransactionCreate(BaseModel):
    location:str
    ammount:float
    user_id:int
    device_id:str

class TransactionResponce(BaseModel):
    id:int
    user_id:int
    location:str
    ammount:float
    device_id:str
    is_fraud:bool

    class Config:
        from_attributes=True