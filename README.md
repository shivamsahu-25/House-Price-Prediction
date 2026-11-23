# House-Price-Prediction
# 🏠 House Price Prediction

A Machine Learning project that predicts house prices in Bengaluru using supervised learning techniques. This project uses data preprocessing, feature engineering, and regression algorithms to estimate property prices based on different housing features.

## 📌 Project Overview

The main objective of this project is to build a machine learning model that predicts house prices based on property-related features such as location, total square footage, number of bedrooms (BHK), bathrooms, and other available attributes.

The project follows a complete machine learning workflow, from data cleaning to model training and saving the trained model for future predictions.

## 🎯 Objectives

- Understand and analyze the Bengaluru house price dataset.
- Perform data cleaning and preprocessing.
- Handle missing values and prepare features.
- Apply supervised machine learning algorithms.
- Train and evaluate regression models.
- Save the trained model using Pickle.
- Prepare the project for integration with a Flask web application.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualization |
| Scikit-learn | Machine learning |
| Pickle | Model serialization |
| Flask | Backend web framework |
| HTML / Bootstrap | Frontend development |
| Git & GitHub | Version control |

## 📂 Project Structure

```text
House_price_pred/
│
├── dataset/
│   ├── Bengaluru_House_Data.csv
│   └── cleaned_data.csv
│
├── model/
│   └── house_price_model_grad.pkl
│
├── templates/
│   └── index.html
│
├── clean_train_model.py
├── main.py
└── README.md
```

## ⚙️ Machine Learning Workflow

1. **Data Collection:** Load the Bengaluru house price dataset.
2. **Data Cleaning:** Handle missing values, duplicates and inconsistent data.
3. **Exploratory Data Analysis:** Understand the dataset and identify patterns.
4. **Feature Engineering:** Prepare relevant features for model training.
5. **Model Training:** Apply supervised learning regression algorithms.
6. **Model Evaluation:** Evaluate model performance using regression metrics.
7. **Model Saving:** Save the trained model using Pickle.

## 📊 Model Evaluation

The project focuses on regression evaluation metrics:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

These metrics help evaluate how accurately the model predicts house prices.

## 🚀 Installation & Setup

**1. Clone the repository**

```bash
git clone https://github.com/shivamsahu-25/House-Price-Prediction.git
```

**2. Navigate to the project directory**

```bash
cd House-Price-Prediction
```

**3. Create a virtual environment**

```bash
python -m venv venv
```

**4. Activate the virtual environment**

Windows:

```bash
venv\Scripts\activate
```

**5. Install dependencies**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn flask
```

**6. Run the project**

```bash
python clean_train_model.py
```

## 🔮 Future Improvements

- Integrate the trained model with Flask.
- Develop an interactive web interface for house price prediction.
- Improve model performance through hyperparameter tuning.
- Add more regression algorithms and compare their results.
- Deploy the application online.

## 👨‍💻 Author

**Shivam Sahu**

- GitHub: [shivam-25-JNCT](https://github.com/shivam-25-JNCT)
- LinkedIn: [Shivam Sahu](https://www.linkedin.com/in/shivam-sahu-540773294/)

---

⭐ If you find this project useful, consider giving it a star!