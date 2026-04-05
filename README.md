# Multimodal AI Hazard Risk Classification System

## Project Overview

This project focuses on building an intelligent system for hazard risk classification in construction and industrial environments using computer vision. 

Unlike traditional object detection systems, this project goes beyond identifying objects (e.g., helmet, person) and instead infers safety risk levels based on contextual understanding.

The system:
- Detects safety-related objects using YOLOv8
- Extracts structured features from detections
- Predicts **risk levels (Low, Medium, High)** using a learned classifier
- Provides **visual grounding** by highlighting hazardous regions

---

##  Problem Statement

In real-world safety-critical environments, detecting objects alone is insufficient. For example, detecting a worker and a helmet separately does not indicate whether the worker is safe.

This project aims to bridge that gap by:
- Modeling **contextual risk**
- Providing **interpretable outputs**
- Enabling **deployable safety monitoring systems**

---

##  Project Structure
Hazard Risk
│
├── data/ # Sample dataset or references
├── notebooks/ # Jupyter notebooks (setup & experiments)
├── src/ # Core pipeline code (data, model, utils)
├── ui/ # Interface (FastAPI / Streamlit - upcoming)
├── results/ # Outputs, plots, and visualizations
├── docs/ # Architecture diagrams and assets
│
├── requirements.txt
└── README.md

## Installation & Setup

### 1. Clone the repository

git clone https://github.com/MeghaSuhanth23/Multimodal-AI-Hazard-Risk-Classification-System.git
cd Hazard Risk

### 2. Install dependencies

pip install -r requirements.txt

### Run Jupyter Notebook

jupyter notebook

### Dataset Information

Dataset: PPE Detection Dataset (YOLOv8 format)
Source: Kaggle (via KaggleHub)
Type: Image dataset with bounding box annotations

**Classes Used**
Person
Helmet
Vest
**Format**
Labels: YOLO format

### Model Pipeline

Input Image → Object Detection (YOLOv8) → Feature Extraction → Risk Classification (MLP / Logistic Regression) → Risk Output (Low / Medium / High)

### Pipeline Workflow
- **Input Image**: User uploads a site photo.

- **Object Detection**: YOLOv8 identifies persons, helmets, and vests.

- **Feature Extraction**: System determines spatial relationships (e.g., is the helmet on the head?).

- **Risk Classification**: MLP / Logistic Regression assigns a risk score.

- **Output**: A labeled image and risk level (Low/Medium/High) are displayed.

### Future Roadmap (MLOps)
The system is designed for future integration with:

- **FastAPI**: For high-performance inference.

- **Docker & Docker Compose**: For seamless containerized deployment.

- **Prometheus & Grafana**: For real-time monitoring of detection accuracy and latency.


### Responsible AI Considerations
- Model may exhibit bias due to dataset limitations
- Predictions are advisory and not for automated enforcement
- Visual grounding improves transparency and interpretability

### Author

**Megha Suhanth Royal Sarvepalli**
**Email**: msarvepalli@ufl.edu

### Project Status
- Initial setup complete
- Model training in progress
- MLOps integration planned
