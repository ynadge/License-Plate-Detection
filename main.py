import streamlit as st
import torch
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import pathlib
import easyocr
import cv2
import numpy as np
from video_prediction import runVideo

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

CFG_MODEL_PATH = "RGBmodel_30epochs/weights/best.pt"
CFG_ENABLE_URL_DOWNLOAD = False
CFG_ENABLE_VIDEO_PREDICTION = True
# End of Configurations

def imageInput(model):
        # Uploading the image.
        image_file = st.file_uploader(
            "Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:  # First column to display uploaded image.
                st.image(img, caption='Uploaded Image',
                         use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join(
                'data/outputs', os.path.basename(imgpath))  # To store prediction image.
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            with st.spinner(text="Predicting..."):
                # YOLOv5 model to detect license plate
                pred = model(imgpath)
                pred.render()
                # save output to file
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(outputpath)
                # OCR model to read license plate
                reader = easyocr.Reader(["en"])
                result = reader.readtext(img)
                text = " ".join([res[1] for res in result])

            # Predictions
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption= "Number Plate: " + text,
                         use_column_width='always')

def videoInput(model):
        uploaded_video = st.file_uploader(
            "Upload A Video", type=['mp4', 'mpeg', 'mov'])
        pred_view = st.empty()
        warning = st.empty()
        if uploaded_video != None:

            # Save video to disk
            ts = datetime.timestamp(datetime.now())  # timestamp a upload
            uploaded_video_path = os.path.join(
                'data/uploads', str(ts)+uploaded_video.name)
            with open(uploaded_video_path, mode='wb') as f:
                f.write(uploaded_video.read())

            # Display uploaded video
            with open(uploaded_video_path, 'rb') as f:
                video_bytes = f.read()
            st.video(video_bytes)
            st.write("Uploaded Video")
            submit = st.button("Run Prediction")
            if submit:
                runVideo(model, uploaded_video_path, pred_view, warning)
																									
def main():
    if CFG_ENABLE_URL_DOWNLOAD:
        downloadModel()
        
    else:
        if not os.path.exists(CFG_MODEL_PATH):
            st.error(
                'Model not found, please config if you wish to download model from url set `cfg_enable_url_download = True`  ', icon="⚠️")

    if CFG_ENABLE_VIDEO_PREDICTION:
        option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
    else:
        option = st.sidebar.radio("Select input type.", ['Image'])
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", [
                                        'cpu', 'cuda'], disabled=False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", [
                                        'cpu', 'cuda'], disabled=True, index=0)
    # -- End of Sidebar

    st.header('📦 YOLOv5 License Plate Detection')

    if option == "Image":
        imageInput(loadmodel(deviceoption))
    elif option == "Video":
        videoInput(loadmodel(deviceoption))

# Downlaod Model from url.
@st.cache_resource
def downloadModel():
    if not os.path.exists(CFG_MODEL_PATH):
        wget.download(url, out="models/")
        

@st.cache_resource
def loadmodel(device):
    if CFG_ENABLE_URL_DOWNLOAD:
        CFG_MODEL_PATH = f"models/{url.split('/')[-1:][0]}"
    model = torch.hub.load('ultralytics/yolov5', 'custom', path="RGBmodel_30epochs/weights/best.pt", force_reload=True, device=device)
    return model


if __name__ == '__main__':
    main()