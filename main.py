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
from schema import IdentityCard, DrivingLicense

import cv2
from paddleocr import PaddleOCR, draw_ocr
import numpy as np
import base64

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
    res = IdentityCard(id="",birth="",name="", imgblur="")
    readfile = await file.read()
    file.filename = f"{uuid.uuid4()}.jpg"
    img = Image.open(io.BytesIO(readfile))
    imgblur = cv2.imdecode(np.frombuffer(readfile, np.uint8), cv2.IMREAD_COLOR)
    h, w = imgblur.shape[:2]
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    img = ImageOps.exif_transpose(img)
    model_path = "detectCCCD.pt"
    model = YOLO(model_path)
    results = model.predict(source=img)
    boxes = results[0].boxes
    crop = ["","",""]
    newcrop = ["","",""]
    for box in boxes:
        imgcrop = img.crop(box=box.xyxy[0].tolist())
        if(int(box.cls[0].tolist()) == 6):
            crop[0] = imgcrop
            newcrop[0] = "imgcrop"
        elif(int(box.cls[0].tolist()) == 8):
            crop[1] = imgcrop
            newcrop[1] = "imgcrop"
        elif(int(box.cls[0].tolist()) == 1):
            crop[2] = imgcrop
            newcrop[2] = "imgcrop"
        if(int(box.cls[0].tolist()) != 11):
            imgblurcrop = imgblur[int(box.xyxy[0].tolist()[1]):int(box.xyxy[0].tolist()[3]), int(box.xyxy[0].tolist()[0]):int(box.xyxy[0].tolist()[2])]
            imgblurcrop = cv2.GaussianBlur(imgblurcrop,(kernel_width, kernel_height), 0)
            imgblur[int(box.xyxy[0].tolist()[1]):int(box.xyxy[0].tolist()[3]), int(box.xyxy[0].tolist()[0]):int(box.xyxy[0].tolist()[2])] = imgblurcrop
    
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
    _, im_arr = cv2.imencode('.jpg', imgblur)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    res.id = s[0]
    res.name = s[1]
    res.birth = s[2]
    res.imgblur = im_b64
    return res

@app.post("/licensePlate")
async def detectLicensePlate(file : UploadFile = File(...)):
    file.filename = f"{uuid.uuid4()}.jpg"
    img = cv2.imdecode(np.frombuffer(await file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    model_path = "biensoxe.pt"
    model = YOLO(model_path)
    results = model.predict(source=img)
    boxes = results[0].boxes
    box = boxes[0]
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    imgcrop = img[int(box.xyxy[0].tolist()[1]):int(box.xyxy[0].tolist()[3]), int(box.xyxy[0].tolist()[0]):int(box.xyxy[0].tolist()[2])]
    assert not isinstance(imgcrop,type(None))   
    result = ocr.ocr(imgcrop, cls=True)
    response = ""
    for idx in range(len(result)):
        res = result[idx]
    for line in res:
        response += " " + line[1][0]
    return {
        "licensePlate" : response
    }
    
@app.post("/DrivingLicense")
async def detectDrivingLicense(file : UploadFile = File(...)):
    res = DrivingLicense(id="",birth="",name="", imgblur="")
    file.filename = f"{uuid.uuid4()}.jpg"
    readfile = await file.read()
    img = Image.open(io.BytesIO(readfile))
    img = ImageOps.exif_transpose(img)
    imgblur = cv2.imdecode(np.frombuffer(readfile, np.uint8), cv2.IMREAD_COLOR)
    h, w = imgblur.shape[:2]
    kernel_width = (w // 7) | 1
    kernel_height = (h // 7) | 1
    model_path = "drivingLicense.pt"
    model = YOLO(model_path)
    results = model.predict(source=img)
    boxes = results[0].boxes
    crop = ["","",""]
    newcrop = ["","",""]
    for box in boxes:
        imgcrop = img.crop(box=box.xyxy[0].tolist())
        if(int(box.cls[0].tolist()) == 6):
            crop[0] = imgcrop
            newcrop[0] = "imgcrop"
        elif(int(box.cls[0].tolist()) == 4):
            crop[1] = imgcrop
            newcrop[1] = "imgcrop"
        elif(int(box.cls[0].tolist()) == 1):
            crop[2] = imgcrop
            newcrop[2] = "imgcrop"
        imgblurcrop = imgblur[int(box.xyxy[0].tolist()[1]):int(box.xyxy[0].tolist()[3]), int(box.xyxy[0].tolist()[0]):int(box.xyxy[0].tolist()[2])]
        imgblurcrop = cv2.GaussianBlur(imgblurcrop,(kernel_width, kernel_height), 0)
        imgblur[int(box.xyxy[0].tolist()[1]):int(box.xyxy[0].tolist()[3]), int(box.xyxy[0].tolist()[0]):int(box.xyxy[0].tolist()[2])] = imgblurcrop
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
    _, im_arr = cv2.imencode('.jpg', imgblur)
    im_bytes = im_arr.tobytes()
    im_b64 = base64.b64encode(im_bytes)
    res.id = s[0]
    res.name = s[1]
    res.birth = s[2]
    res.imgblur = im_b64
    return res