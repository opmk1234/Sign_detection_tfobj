# ✋ Real-Time Hand Sign Detection using TensorFlow Object Detection API

This project implements a **real-time hand sign detection system** using **TensorFlow Object Detection API** and **SSD MobileNet V2 FPNLite (320×320)**.
It supports **training, evaluation, real-time webcam inference**, and **exporting models** to **SavedModel, TFJS, and TFLite**.

---

## 🚀 Features

* 📷 Webcam image capture using OpenCV
* 🏷️ Manual image labeling with `labelImg`
* 🧠 Transfer learning with SSD MobileNet V2
* 🎥 Real-time hand sign detection
* 📦 Export models for:

  * TensorFlow SavedModel
  * TensorFlow.js (browser)
  * TensorFlow Lite (Raspberry Pi / Edge devices)

---

## ✋ Detected Hand Signs

| Label      |
| ---------- |
| ThumbsUp   |
| ThumbsDown |
| Thankyou   |
| LiveLong   |

---

## 🧱 Project Structure

```
Tensorflow/
├── workspace/
│   ├── images/
│   │   ├── collectedimages/
│   │   ├── train/
│   │   └── test/
│   ├── annotations/
│   │   ├── label_map.pbtxt
│   │   ├── train.record
│   │   └── test.record
│   ├── models/
│   │   └── my_ssd_mobnet/
│   └── pre-trained-models/
├── models/
│   └── research/
├── scripts/
└── protoc/
```

---

## 🧪 Environment (⚠️ DO NOT UPGRADE)

This project is **very version-sensitive**.
Use **exactly** the following stack:

```
Python              3.9.x
TensorFlow          2.12.0
NumPy               1.23.5
protobuf            3.20.3
protoc              3.20.x
tensorflow-io       0.31.0
tensorflow-addons   0.20.0
tf-models-official  2.12.0 (NO DEPS)
opencv-python
matplotlib
lxml
PyYAML
```

> ❌ Upgrading TensorFlow, protobuf, NumPy, or Keras **WILL BREAK** the pipeline.

---

## 🛠️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install tensorflow==2.12.0
pip install numpy==1.23.5 protobuf==3.20.3
pip install tensorflow-addons==0.20.0 tensorflow-io==0.31.0
pip install --no-deps tf-models-official==2.12.0
pip install opencv-python matplotlib lxml pyyaml pillow cython contextlib2 tqdm
```

---

### 4️⃣ Install TensorFlow Object Detection API

```bash
git clone https://github.com/tensorflow/models Tensorflow/models
cd Tensorflow/models/research
protoc object_detection/protos/*.proto --python_out=.
pip install .
pip install -e slim
```

Verify:

```bash
python object_detection/builders/model_builder_tf2_test.py
```

---

## 📷 Collect Images

```python
cap = cv2.VideoCapture(0)
```

* Images are saved under:

```
Tensorflow/workspace/images/collectedimages/<label>/
```

---

## 🏷️ Label Images

```bash
cd Tensorflow/lableimg
python labelImg.py
```

* Save annotations in **Pascal VOC (`.xml`)**
* Images and XML must be in the same folder

---

## 🧾 Create Label Map

```text
item {
  name: "ThumbsUp"
  id: 1
}
item {
  name: "ThumbsDown"
  id: 2
}
item {
  name: "Thankyou"
  id: 3
}
item {
  name: "LiveLong"
  id: 4
}
```

Saved as:

```
Tensorflow/workspace/annotations/label_map.pbtxt
```

---

## 📦 Generate TFRecords

```bash
python generate_tfrecord.py \
  -x Tensorflow/workspace/images/train \
  -l Tensorflow/workspace/annotations/label_map.pbtxt \
  -o Tensorflow/workspace/annotations/train.record
```

Repeat for `test.record`.

---

## 🧠 Training the Model

```bash
python Tensorflow/models/research/object_detection/model_main_tf2.py \
  --model_dir=Tensorflow/workspace/models/my_ssd_mobnet \
  --pipeline_config_path=Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config \
  --num_train_steps=2000
```

---

## 🎥 Real-Time Detection

```python
cap = cv2.VideoCapture(0)
```

* Press **`q`** to quit

---

## 📤 Export Model

### SavedModel

```bash
python exporter_main_v2.py \
  --input_type=image_tensor \
  --pipeline_config_path=pipeline.config \
  --trained_checkpoint_dir=models/my_ssd_mobnet \
  --output_directory=export
```

---

### TensorFlow.js

```bash
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_graph_model \
  export/saved_model tfjsexport
```

---

### TensorFlow Lite (⚠️ Important)

⚠️ **Model input size MUST be 320×320**

```bash
tflite_convert \
  --saved_model_dir=tfliteexport/saved_model \
  --output_file=detect.tflite \
  --input_shapes=1,320,320,3 \
  --input_arrays=normalized_input_image_tensor \
  --output_arrays=TFLite_Detection_PostProcess,\
TFLite_Detection_PostProcess:1,\
TFLite_Detection_PostProcess:2,\
TFLite_Detection_PostProcess:3 \
  --allow_custom_ops
```

---

## ⚠️ Common Issues

| Issue                     | Fix                           |
| ------------------------- | ----------------------------- |
| DNS / `contextlib2` error | Use `pip install contextlib2` |
| TensorFlow Addons warning | Ignore (expected)             |
| Model crashes on TFLite   | Fix input size to **320×320** |
| Training unstable         | Add more images per class     |

---

## 📌 Notes

* Dataset should be **balanced**
* Minimum **100+ images per class** recommended
* SSD MobileNet is optimized for **real-time performance**

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

* TensorFlow Object Detection API
* SSD MobileNet V2
* OpenCV
* labelImg

---

## ⭐ If this helped you

Star ⭐ the repo and feel free to fork!

---

**Happy Detecting! 🚀**
