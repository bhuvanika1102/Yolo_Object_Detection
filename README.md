# YOLOv8 Object Detection on Colorful Fashion Dataset

This project demonstrates object detection using the YOLOv8 model from the Ultralytics library, applied to images from a colorful fashion dataset. The model detects multiple object classes in real-time with high accuracy and visualizes results using matplotlib.

## 🚀 Features

- Uses **YOLOv8 (medium)** pre-trained model.
- Predicts object classes in fashion-related images.
- Visualizes detection output using matplotlib.
- Saves output to `runs/detect/predict`.

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/yolo-object-detection.git
   cd yolo-object-detection
2. Install dependencies:
    ```bash
    pip install ultralytics matplotlib opencv-python tqdm
## 📁 Dataset
Place your images and annotations in the following structure:
    
    project-root/
    │
    ├── colorful_fashion_dataset_for_object_detection/
    │   ├── JPEGImages/
    │   │   └── *.jpg
    │   ├── Annotations_txt/
    │       └── *.txt
    Sample image used: GRdCCA.jpg

## 🧠 Model Used
YOLOv8m: A medium-sized YOLOv8 model pretrained on COCO.

You can also try yolov8n-face.pt or other model variants.

## 🧪 Running the Notebook
1. Open the yolo_object_detection.ipynb file.

2. Run all cells step by step.

3. Detected objects are displayed inline using matplotlib and saved to runs/detect/predict.

Example prediction:

    from ultralytics import YOLO
    model = YOLO("yolov8m.pt")
    result = model.predict(source='GRdCCA.jpg', conf=0.1, save=True)

## 🖼️ Output
The output image with bounding boxes will be saved in:

    runs/detect/predict/GRdCCA.jpg
    
## 📦 Requirements
Python 3.7+
torch
ultralytics
opencv-python
matplotlib

## 📌 Notes
Make sure yolov8m.pt is downloaded or available locally.
For GPU acceleration, ensure PyTorch is installed with CUDA.

## Contact
For any questions or comments, please reach out bhuvani1102@gmail.com.
