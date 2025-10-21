import os, numpy as np, pandas as pd, tensorflow as tf, tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from data_loader import load_video, load_image

MODEL_PATH = "model/video_repair_estimator.h5"

class USELayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder = hub.KerasLayer(
            "https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False
        )

    def call(self, inputs):
        return self.encoder(inputs)


def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        print("Trained model not found!")
        return None

    model = load_model(
        MODEL_PATH,
        custom_objects={"USELayer": USELayer, "KerasLayer": hub.KerasLayer}
    )
    return model

def demo_predict(model, sample_row):
    video = np.expand_dims(load_video(sample_row["video_path"]), 0)
    image = np.expand_dims(load_image(sample_row["image_path"]), 0)
    text = np.array([sample_row["description"]])
    meta = np.array([[sample_row.get("asset_type_enc",0), sample_row.get("humidity",0.0), sample_row.get("temperature",0.0)]])

    preds = model.predict({"video_input": video, "image_input": image, "text_input": text, "meta_input": meta})
    cost_time = preds[0][0]
    material = np.argmax(preds[1][0])
    print("=== Demo Prediction ===")
    print("Estimated cost: ", round(float(cost_time[0]), 2))
    print("Estimated time (days): ", round(float(cost_time[1]), 2))
    print("Predicted material class index: ", int(material))

def main():
    model = load_trained_model()
    if model:
        print("Model loaded successfully. Ready for prediction.")

if __name__ == "__main__":
    main()