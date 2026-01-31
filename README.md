# ✅ Final README.md (copy & paste)

```markdown
# Multi-Class Image Classifier using Transfer Learning

This project implements an end-to-end multi-class image classification pipeline using transfer learning and exposes the trained model through a REST API.

The workflow includes data preprocessing, model training, evaluation, and deployment using Docker.

---

## Project Structure

```

.
├── data/
│   ├── train/
│   └── val/
├── model/
│   └── image_classifier.pth
├── results/
│   └── metrics.json
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── api.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md

````

---

## Dataset

The Caltech-101 dataset is used.  
A preprocessing script downloads the dataset, selects 10 classes, and splits the data into training and validation sets (80/20).

---

## Model

A pre-trained ResNet-18 model is used for transfer learning.

The final fully-connected layer is replaced to match the number of classes.  
Only the classifier layer is trained while the backbone is frozen.

---

## Data Augmentation

The training pipeline applies the following augmentations:

- RandomResizedCrop
- RandomHorizontalFlip
- RandomRotation

---

## Training

Run:

```bash
py -3.11 src/train.py
````

The trained model is saved to:

```
model/image_classifier.pth
```

---

## Evaluation

Run:

```bash
py -3.11 src/evaluate.py
```

The evaluation results are stored in:

```
results/metrics.json
```

The metrics include:

* accuracy
* weighted precision
* weighted recall
* confusion matrix

---

## API

The REST API is built using FastAPI.

### Health check

```
GET /health
```

Response:

```json
{"status":"ok"}
```

### Prediction

```
POST /predict
```

Request:

* multipart/form-data
* key: `file`

Response:

```json
{
  "predicted_class": "class_name",
  "confidence": 0.0
}
```

---

## Docker Deployment

Create a `.env` file from the provided template:

```
API_PORT=8000
MODEL_PATH=model/image_classifier.pth
```

Build and run:

```bash
docker compose up --build
```

---

## Example request

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@data/val/Leopards/1009.jpg"
```

---

## Tools and Libraries

* Python
* PyTorch
* Torchvision
* FastAPI
* Scikit-Learn
* Docker

```

