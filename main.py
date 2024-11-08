from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.responses import RedirectResponse
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


@app.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")
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
    crop = ["","",""]
    newcrop = ["","",""]
    for box in boxes:
        imgcrop = img.crop(box=box.xyxy[0].tolist())
        if(int(box.cls[0].tolist()) == 0):
            crop[0] = imgcrop
            newcrop[0] = "imgcrop"
        elif(int(box.cls[0].tolist()) == 1):
            crop[1] = imgcrop
            newcrop[1] = "imgcrop"
        elif(int(box.cls[0].tolist()) == 2):
            crop[2] = imgcrop
            newcrop[2] = "imgcrop"
    config = Cfg.load_config_from_file("config/base.yml")
    detector = Predictor(config)
    if(crop.count("") == 3):
        raise HTTPException(status_code=404, detail="Item not found") 
    index = -1
    while(crop.__contains__("")):
        index = crop.index("")
        crop[index] = crop[0]
    s = detector.predict_batch(crop)
    if(index != -1):
        while(newcrop.__contains__("")):
            index = newcrop.index("")
            s[index] = ""
            newcrop.remove("")
    res.id = s[0]
    res.name = s[1]
    res.birth = s[2]
    return res