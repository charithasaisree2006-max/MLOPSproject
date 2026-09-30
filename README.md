# YouTube Views Prediction using Machine Learning

## 📌 Project Overview

This project is a Machine Learning pipeline developed to predict the number of views of YouTube videos.

The project demonstrates an end-to-end Machine Learning workflow including:

- Data validation
- Data preprocessing
- Feature preparation
- Model training
- Model evaluation
- Model saving
- MLflow experiment tracking
- Prediction generation
- Reproducible project structure

---

## 🎯 Objective

The main objective of this project is to build a Machine Learning model that can learn patterns from historical YouTube video data and predict video views.

The project also demonstrates how MLOps concepts can be applied to a Machine Learning project using MLflow.

---

## 📊 Dataset

The project uses a YouTube video dataset containing information about videos and their performance.

The dataset includes information such as:

- Video details
- Channel information
- Category information
- Views
- Likes
- Comments
- Other video-related attributes

The raw dataset is stored inside the `data/raw/` directory.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- MLflow
- Jupyter Notebook

---

## 📁 Project Structure

```text
project/
│
├── artifacts/
│   ├── best_model_summary.json
│   └── test_predictions.csv
│
├── data/
│   ├── processed/
│   └── raw/
│       ├── IN_category_id.json
│       └── INvideos.csv
│
├── models/
│   ├── best_model.pkl
│   └── preprocessing_info.pkl
│
├── notebooks/
│
├── outputs/
│
├── pipelines/
│
├── src/
│   ├── preprocess.py
│   ├── preprocess_pipeline.py
│   ├── train.py
│   ├── train_mlflow.py
│   └── validate_data.py
│
├── .gitignore
├── README.md
└── requirements.txt