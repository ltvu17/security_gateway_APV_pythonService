from pydantic import BaseModel
class IdentityCard(BaseModel):
    id: str
    name: str
    birth: str