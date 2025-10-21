import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_hub as hub
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data_loader import load_video, load_image

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs detected:", gpus)
else:
    print("No GPU detected - ensure TF-GPU is installed for GPU training.")

DATA_CSV = "data/dataset.csv"
IMG_SIZE = (224, 224)
NUM_FRAMES = 16

def build_model(num_material_classes):
    video_input = keras.Input(shape=(NUM_FRAMES, IMG_SIZE[0], IMG_SIZE[1], 3), name="video_input")
    image_input = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image_input")
    text_input = keras.Input(shape=(), dtype=tf.string, name="text_input")
    meta_input = keras.Input(shape=(3,), name="meta_input")

    
    x = layers.Conv3D(32, (3,3,3), activation="relu", padding="same")(video_input)
    x = layers.MaxPooling3D((1,2,2))(x)
    x = layers.Conv3D(64, (3,3,3), activation="relu", padding="same")(x)
    x = layers.MaxPooling3D((2,2,2))(x)
    x = layers.GlobalAveragePooling3D()(x)
    video_feat = layers.Dense(128, activation="relu")(x)

    
    base_cnn = keras.applications.EfficientNetB0(include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1],3), pooling="avg", weights=None)
    base_cnn.trainable = False
    img_feat = base_cnn(image_input)
    img_feat = layers.Dense(128, activation="relu")(img_feat)

    
    class USELayer(layers.Layer):
        def __init__(self, **kwargs): 
            super().__init__(**kwargs)
            self.encoder = hub.KerasLayer(
                "https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False
            )

        def call(self, inputs):
            return self.encoder(inputs)
            
    txt_feat = USELayer()(text_input)
    txt_feat = layers.Dense(128, activation="relu")(txt_feat)
    
    meta_feat = layers.Dense(32, activation="relu")(meta_input)

   
    combined = layers.Concatenate()([video_feat, img_feat, txt_feat, meta_feat])
    combined = layers.Dense(256, activation="relu")(combined)
    combined = layers.Dropout(0.3)(combined)

   
    cost_time = layers.Dense(2, activation="linear", name="cost_time_output")(combined)
    material = layers.Dense(num_material_classes, activation="softmax", name="material_output")(combined)

    model = keras.Model(inputs=[video_input, image_input, text_input, meta_input], outputs=[cost_time, material])
    
    
    model.compile(optimizer=keras.optimizers.Adam(1e-4),
                  loss={
                      "cost_time_output": keras.losses.MeanSquaredError(),
                      "material_output": keras.losses.CategoricalCrossentropy()
                  },
                  metrics={
                      "cost_time_output": keras.metrics.MeanAbsoluteError(), 
                      "material_output": keras.metrics.CategoricalAccuracy()
                  })
    return model

def main():
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"{DATA_CSV} not found.")

    df = pd.read_csv(DATA_CSV)
    le_asset = LabelEncoder()
    df["asset_type_enc"] = le_asset.fit_transform(df["asset_type"].astype(str))
    le_mat = LabelEncoder()
    df["material_enc"] = le_mat.fit_transform(df["material"].astype(str))
    num_material_classes = len(le_mat.classes_)

    scaler = StandardScaler()
    df[["humidity", "temperature"]] = scaler.fit_transform(df[["humidity", "temperature"]].fillna(0.0))

    print("Loading media into memory (demo).")
    videos = np.stack([load_video(p, num_frames=NUM_FRAMES, resize=IMG_SIZE) for p in df["video_path"]])
    images = np.stack([load_image(p, IMG_SIZE) for p in df["image_path"]])
    texts = df["description"].astype(str).values
    metadata = df[["asset_type_enc", "humidity", "temperature"]].values
    targets = df[["cost", "time_days"]].values
    material_target = tf.keras.utils.to_categorical(df["material_enc"], num_classes=num_material_classes)

    model = build_model(num_material_classes)
    model.summary()

    model.fit({"video_input": videos, "image_input": images, "text_input": texts, "meta_input": metadata},
              {"cost_time_output": targets, "material_output": material_target},
              validation_split=0.2, epochs=5, batch_size=2)

    os.makedirs("model", exist_ok=True)
    model.save("model/video_repair_estimator.h5")
    print("Saved model/video_repair_estimator.h5")

if __name__ == "__main__":
    main()