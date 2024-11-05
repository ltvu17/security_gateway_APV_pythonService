FROM python:3.9.8-slim
WORKDIR /app
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6
COPY . /app
RUN pip install -r requirements.txt