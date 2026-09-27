\# 🛫 British Airways Data Science Job Simulation (Forage)



<div align="center">



!\[British Airways](https://img.shields.io/badge/British%20Airways-Data%20Science-075AAA?style=for-the-badge\&logo=britishairways\&logoColor=white)

!\[Forage](https://img.shields.io/badge/Forage-Job%20Simulation-00A4B4?style=for-the-badge)

!\[Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge\&logo=python\&logoColor=white)

!\[scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge\&logo=scikit-learn\&logoColor=white)



\*\*Predicting Customer Buying Behaviour — A Machine Learning Approach to Flight Booking Conversion\*\*



\[Overview](#-overview) • \[Dataset](#-dataset) • \[Workflow](#-workflow) • \[Results](#-results) • \[Installation](#-installation)



</div>



\---



\## 📖 Overview



This repository contains my submission for the \*\*British Airways Data Science Job Simulation\*\* hosted on \*\*Forage\*\*. The simulation replicates real-world tasks performed by British Airways' data science team, focusing on understanding customer behavior in the aviation industry.



The core objective is to build a machine learning model that predicts whether a customer will \*\*complete a booking\*\* (`booking\_complete`) after performing a flight search. By accurately predicting this behavior, airlines can:



\- 🎯 Target promotions more effectively

\- 💰 Optimize pricing strategies

\- 📈 Increase conversion rates

\- ✈️ Enhance the overall customer experience



\---



\## 📊 Dataset



The dataset (`customer\_booking.csv`) contains \*\*50,000 records\*\* of flight searches performed by customers, with \*\*14 features\*\*:



| Feature | Description |

|---------|-------------|

| `num\_passengers` | Number of passengers |

| `sales\_channel` | Sales channel (Internet/Mobile) |

| `trip\_type` | Trip type (RoundTrip, OneWay, CircleTrip) |

| `purchase\_lead` | Days between booking and flight |

| `length\_of\_stay` | Length of stay (days) |

| `flight\_hour` | Flight hour |

| `flight\_day` | Flight day |

| `route` | Flight route (Origin-Destination Code) |

| `booking\_origin` | Country of origin of the booking |

| `wants\_extra\_baggage` | Wants to purchase extra baggage (1/0) |

| `wants\_preferred\_seat` | Wants to choose a preferred seat (1/0) |

| `wants\_in\_flight\_meals` | Wants to order in-flight meals (1/0) |

| `flight\_duration` | Flight duration (hours) |

| \*\*`booking\_complete`\*\* | \*\*Target (1 = Booking Complete, 0 = Not Complete)\*\* |



> ⚠️ \*\*Note:\*\* The dataset has a class imbalance — only \*\*\~15%\*\* of searches result in a completed booking.



\---



\## 🔄 Workflow



\### 1️⃣ Data Exploration

\- Loaded and inspected the dataset structure (shape, dtypes, head)

\- Performed \*\*univariate analysis\*\* on numerical and categorical variables

\- Analyzed target distribution and correlation matrix



\### 2️⃣ Data Preparation

\- ✅ \*\*Missing Values:\*\* None detected across all 14 columns

\- ✅ \*\*Outlier Detection:\*\* Identified outliers in `purchase\_lead`, `length\_of\_stay`, and `flight\_duration` using IQR method



\### 3️⃣ Feature Engineering

\- \*\*One-Hot Encoding\*\* for low-cardinality features (`sales\_channel`, `trip\_type`, `flight\_day`)

\- \*\*Frequency Encoding\*\* for high-cardinality features (`route` — 799 unique, `booking\_origin` — 104 unique)

\- \*\*Created new features:\*\*

&#x20; - `total\_wants` — Sum of extra baggage + preferred seat + in-flight meals

&#x20; - `lead\_duration\_ratio` — `purchase\_lead / (flight\_duration + 1)`



\### 4️⃣ Modeling

\- \*\*Algorithm:\*\* `RandomForestClassifier` (100 estimators, `class\_weight='balanced'`)

\- \*\*Validation:\*\* 5-Fold Stratified Cross-Validation

\- \*\*Train/Test Split:\*\* 80/20 with stratification

\- \*\*Scaling:\*\* `StandardScaler` applied



\### 5️⃣ Evaluation

\- Confusion Matrix, ROC Curve, Feature Importance, CV Scores

\- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC



\---



\## 📈 Results



\### 🎯 Cross-Validation (5-Fold Stratified)

Scores per Fold: \[0.7509, 0.7437, 0.7546, 0.7579, 0.7574]

Mean ROC-AUC : 0.7529 (± 0.0053)



text



\### 🧪 Test Set Performance



| Metric | Score |

|--------|-------|

| \*\*Accuracy\*\* | 0.8537 |

| \*\*Precision\*\* | 0.5797 |

| \*\*Recall\*\* | 0.0802 |

| \*\*F1-Score\*\* | 0.1409 |

| \*\*ROC-AUC\*\* | \*\*0.7714\*\* |



\### 🔝 Top 10 Feature Importances



| Rank | Feature | Importance |

|------|---------|------------|

| 1 | `origin\_freq` | 0.1475 |

| 2 | `lead\_duration\_ratio` | 0.1292 |

| 3 | `route\_freq` | 0.1272 |

| 4 | `purchase\_lead` | 0.1214 |

| 5 | `length\_of\_stay` | 0.1098 |

| 6 | `flight\_hour` | 0.1046 |

| 7 | `flight\_duration` | 0.0710 |

| 8 | `num\_passengers` | 0.0364 |

| 9 | `total\_wants` | 0.0228 |

| 10 | `flight\_day\_Mon` | 0.0145 |



\### 📉 Key Insights



\- \*\*`origin\_freq`\*\* (booking origin popularity) is the strongest predictor of booking completion

\- \*\*`lead\_duration\_ratio`\*\* — a custom engineered feature — ranked #2, proving feature engineering value

\- The model achieves a solid \*\*ROC-AUC of 0.77\*\*, well above the 0.5 baseline

\- \*\*Low recall (8%)\*\* indicates room for improvement in catching actual bookings — a common issue with imbalanced data



\---



\## 📸 Visualizations



\### Confusion Matrix

!\[Confusion Matrix](outputs/confusion\_matrix.png)



\### ROC Curve

!\[ROC Curve](outputs/roc\_curve.png)



\### Feature Importance (Top 10)

!\[Feature Importance](outputs/feature\_importance.png)



\### Cross-Validation Scores

!\[CV Scores](outputs/cv\_scores.png)



\### All Feature Importances

!\[All Feature Importances](outputs/feature\_importance\_all.png)



\---



\## 🗂️ Repository Structure

british-airways-data-science-forage/

│

├── 📓 Predicting\_Customer\_Buying\_Behaviour.ipynb # Main notebook

├── 📄 customer\_booking.csv # Dataset (50K records)

├── 📄 README.md # Documentation

├── 📄 requirements.txt # Dependencies

├── 📄 LICENSE # MIT License

│

└── 📊 outputs/

├── confusion\_matrix.png # Confusion Matrix

├── roc\_curve.png # ROC Curve (AUC visualization)

├── feature\_importance.png # Top 10 Features

├── cv\_scores.png # Cross-Validation Scores

└── feature\_importance\_all.png # All Features



text



\---



\## ⚙️ Installation



\### Prerequisites

\- Python 3.8+

\- pip



\### Setup



```bash

\# Clone the repository

git clone https://github.com/Nebukhadnezar/british-airways-data-science-forage.git

cd british-airways-data-science-forage



\# Create a virtual environment (optional but recommended)

python -m venv venv



\# Activate (Windows)

venv\\Scripts\\activate



\# Activate (Linux/Mac)

source venv/bin/activate



\# Install dependencies

pip install -r requirements.txt

Run the Notebook

bash

jupyter notebook Predicting\_Customer\_Buying\_Behaviour.ipynb

🛠️ Tech Stack

Category	Tools

Language	Python 3

Data Manipulation	Pandas, NumPy

Visualization	Matplotlib, Seaborn

Machine Learning	Scikit-learn (Random Forest, StratifiedKFold, StandardScaler)

Environment	Jupyter Notebook / Google Colab

🚀 Future Improvements

□ Try XGBoost / LightGBM for better performance on imbalanced data

□ Apply SMOTE or other resampling techniques to improve recall

□ Hyperparameter tuning with Optuna or GridSearchCV

□ Add SHAP values for model interpretability

□ Deploy the model using FastAPI or Streamlit

🎓 About the Simulation

This project was completed as part of the British Airways Data Science Job Simulation on Forage. The simulation provides practical exposure to real tasks performed by British Airways' data scientists, including:



Task 1: Web scraping customer reviews \& sentiment analysis



Task 2: Predicting customer buying behaviour (this project)



Task 3: Presenting findings to stakeholders



👤 Author

Nebukhadnezar



GitHub: @Nebukhadnezar



📜 License

This project is licensed under the MIT License — see the LICENSE file for details.



Note: The dataset is provided by British Airways via Forage and is intended for educational purposes only.



<div align="center">

⭐ If you find this project helpful, please consider giving it a star!

Made with ❤️ and ☕ for Data Science



</div> ```

