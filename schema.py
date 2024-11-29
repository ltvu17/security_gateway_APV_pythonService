from pydantic import BaseModel
class IdentityCard(BaseModel):
    id: str
    name: str
    birth: str
class DrivingLicense(BaseModel):
    id:str
    name:str
    birth:str
    