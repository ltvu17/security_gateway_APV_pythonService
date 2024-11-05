from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uuid
from PIL import Image
import io
from ultralytics import YOLO
from PIL import Image, ImageOps
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from schema import IdentityCard

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"hello": "haha"}
@app.post("/IdentityCard")
async def detectCCCD(file : UploadFile = File(...)):
    res = IdentityCard(id="",birth="",name="")
    file.filename = f"{uuid.uuid4()}.jpg"
    img = Image.open(io.BytesIO(await file.read()))
    img = ImageOps.exif_transpose(img)
    model_path = "best.pt"
    model = YOLO(model_path)
    results = model.predict(source=img)
    boxes = results[0].boxes
    for box in boxes:
        imgcrop = img.crop(box=box.xyxy[0].tolist())
        config = Cfg.load_config_from_file("config/base.yml")
        detector = Predictor(config)
        s = detector.predict(imgcrop)
        if(int(box.cls[0].tolist()) == 0):
            res.id = s
        elif(int(box.cls[0].tolist()) == 1):
            res.name = s
        elif(int(box.cls[0].tolist()) == 2):
            res.birth = s
    if(res.id == "" and res.birth =="" and res.name == ""):
        raise HTTPException(status_code=404, error="Item not found") 
    return res