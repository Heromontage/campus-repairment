#!/usr/bin/env python3
import cv2, numpy as np
from tensorflow.keras.preprocessing import image as keras_image

def load_video(path, num_frames=16, resize=(224,224)):
    cap = cv2.VideoCapture(path)
    frames = []
    if not cap.isOpened():
        for _ in range(num_frames):
            frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.float32))
        return np.array(frames, dtype=np.float32)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        total_frames = num_frames*2
    step = max(1, total_frames // num_frames)

    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.float32))
            continue
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype("float32") / 255.0)
        for _ in range(step-1):
            cap.grab()
    cap.release()
    return np.array(frames, dtype=np.float32)

def load_image(path, size=(224,224)):
    try:
        img = keras_image.load_img(path, target_size=size)
        arr = keras_image.img_to_array(img).astype("float32") / 255.0
        return arr
    except Exception as e:
        return np.zeros((size[0], size[1], 3), dtype=np.float32)
