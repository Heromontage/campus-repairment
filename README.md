# campus-repairment
# Repair Cost & Material Estimator (Multi-Modal Model)

This project implements a multi-modal deep learning model using TensorFlow/Keras to estimate repair **costs**, **time**, and the required **material class** based on video, image, text (description), and numerical metadata. All serialization and loading issues encountered during debugging have been resolved.

---

## 🚀 Project Structure

The core functionality is split across the following files:

| File/Directory | Description |
| :--- | :--- |
| `train_model.py` | Script to **build, train, and save** the multi-modal neural network model. **(Corrected)** |
| `main_demo.py` | Script to **load the trained model** (with `compile=False` for inference speed) and perform a prediction demonstration. **(Corrected)** |
| `data_loader.py` | Utility functions for pre-processing and loading video and image data. |
| `utils.py` | Utility script to ensure necessary directories (`data`, `model`) exist. |
| `data/` | Directory for all input data, including the CSV and media files. |
| `model/` | Directory where the trained model (`video_repair_estimator.h5`) is saved. |

---

## ⚙️ Setup and Installation

### 1. Environment Setup

It is highly recommended to use a virtual environment:

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate # Windows
````

### 2\. Install Dependencies

Install TensorFlow, TensorFlow Hub, OpenCV, and other required libraries:

```bash
pip install tensorflow tensorflow-hub pandas scikit-learn numpy opencv-python-headless
```

### 3\. Data Preparation

1.  **Create Directories:** Run `utils.py` to set up the expected file structure:

    ```bash
    python utils.py
    ```

2.  **Dataset File:** The `train_model.py` script requires a file at **`data/dataset.csv`**. This file must include columns for input features (`video_path`, `image_path`, `description`, `asset_type`, `humidity`, `temperature`) and target labels (`cost`, `time_days`, `material`).

3.  **Media Files:** Place the video and image files referenced in the CSV into the `data/videos/` and `data/images/` directories, respectively.

