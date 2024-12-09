from pydantic import BaseModel
class IdentityCard(BaseModel):
    id: str
    name: str
    birth: str
    imgblur: str
class DrivingLicense(BaseModel):
    id:str
    name:str
    birth:str
    imgblur: str