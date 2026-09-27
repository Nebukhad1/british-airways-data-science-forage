#!/usr/bin/env python
# coding: utf-8

# # **Project Task 2: Predicting Customer Buying Behaviour**

# **1. Introduction**
# 
# In the aviation industry, understanding customer behavior is crucial for maximizing revenue. One of the most important indicators is whether a customer will complete a booking (booking_complete) after performing a flight search. By predicting this behavior, airlines can target promotions more effectively, optimize pricing strategies, and ultimately increase conversion rates.
# 
# **2. Background**
# 
# This dataset contains information about flight searches performed by customers. Each row represents a single search, with various features such as flight route, booking lead time, baggage preferences, and more. The ultimate goal is to build a machine learning model that can predict whether a search will result in a booking (booking_complete = 1) or not (booking_complete = 0).
# 
# **3. Project Objective**
# 
# The main objectives of this project are:
# 
# 
# *   Data Exploration: Understand the data characteristics, variable distributions, and relationships between features.
# *   Data Preparation: Clean the data, handle missing values, and perform feature engineering to improve model quality.
# *   Modeling: Train a machine learning model (specifically Random Forest) to predict booking_complete
# *   Evaluation: Evaluate the model's performance using cross-validation and relevant evaluation metrics.
# *   Presentation: Summarize findings and insights into a single presentation slide.
# 
# **4. Workflow Overview**
# 
# The workflow for this project is as follows:
# 
# 
# 
# *   Load and Check Data: Load the dataset and examine its initial structure.
# *   Variable Description: Understand the meaning of each column.
# *   Univariate Variable Analysis: Analyze the distribution of each variable individually.
# *   Basic Data Analysis: Look at descriptive statistics and correlations between variables.
# *   Outlier Detection: Detect and handle outliers.
# *   Missing Value: Check and handle missing values.
# *   Visualization: Create charts to understand data patterns.
# *   Feature Engineering: Create new, relevant features.
# *   Modeling: Train a Random Forest model and evaluate its performance.
# 
# **5. Table of Contents**
# 
# 
# 
# *   Load and Check Data
# *   Variable Description
# *   Univariate Variable Analysis
# *   Basic Data Analysis
# *   Outlier Detection
# *   Missing Value
# *   Visualization
# *   Feature Engineering
# *   Modeling
# 
# 
# 
# 
# 

# # **6. Code Step-by-Step**

# **Step 1: Load and Check Data**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('customer_booking.csv', encoding='latin-1') # Using latin-1 encoding for special characters

# View the first 5 rows
print("--- First 5 Rows ---")
print(df.head())

# View general dataset information
print("\n--- Dataset Information ---")
print(df.info())

# View dataset shape
print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")


# **Step 2: Variable Description**

# In[ ]:


# Variable Description
print("--- Variable Description ---")
print("""
num_passengers       : Number of passengers
sales_channel        : Sales channel (Internet/Mobile)
trip_type            : Trip type (RoundTrip, OneWay, CircleTrip)
purchase_lead        : Days between booking and flight
length_of_stay       : Length of stay (days)
flight_hour          : Flight hour
flight_day           : Flight day
route                : Flight route (Origin-Destination Code)
booking_origin       : Country of origin of the booking
wants_extra_baggage  : Wants to purchase extra baggage (1/0)
wants_preferred_seat : Wants to choose a preferred seat (1/0)
wants_in_flight_meals: Wants to order in-flight meals (1/0)
flight_duration      : Flight duration (hours)
booking_complete     : Target (1 = Booking Complete, 0 = Not Complete)
""")


# **Step 3: Univariate Variable Analysis**

# In[ ]:


# Separating numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

print(f"Numerical Columns: {list(numerical_cols)}")
print(f"Categorical Columns: {list(categorical_cols)}")

# Numerical Analysis
print("\n--- Numerical Descriptive Statistics ---")
print(df[numerical_cols].describe().T)

# Categorical Analysis
print("\n--- Categorical Descriptive Statistics ---")
for col in categorical_cols:
    print(f"\nColumn: {col}")
    print(df[col].value_counts().head())


# **Step 4: Basic Data Analysis**

# In[ ]:


# Check correlation between numerical variables
plt.figure(figsize=(12, 8))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Check target distribution
print("\n--- Target Distribution (booking_complete) ---")
print(df['booking_complete'].value_counts(normalize=True))


# **Step 5: Outlier Detection**

# In[ ]:


# Detect outliers using IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Check outliers on key numerical columns
for col in ['purchase_lead', 'length_of_stay', 'flight_duration']:
    outliers, lb, ub = detect_outliers_iqr(df, col)
    print(f"Column {col}: {len(outliers)} outliers (Bounds: {lb:.2f} - {ub:.2f})")

# Visualize Boxplots
plt.figure(figsize=(15, 5))
for i, col in enumerate(['purchase_lead', 'length_of_stay', 'flight_duration']):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(col)
plt.tight_layout()
plt.show()


# **Step 6: Missing Value**

# In[ ]:


# Check for missing values
print("--- Missing Values ---")
print(df.isnull().sum())

# If there are missing values, handle them (e.g., drop or impute)
# df = df.dropna() # Example: drop
# df['column'].fillna(df['column'].median(), inplace=True) # Example: impute with median


# **Step 7: Visualization**

# In[ ]:


# Visualize Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='booking_complete', data=df)
plt.title('Distribution of Booking Complete')
plt.show()

# Visualize Relationship between Categorical Features and Target
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.countplot(x='sales_channel', hue='booking_complete', data=df, ax=axes[0,0])
sns.countplot(x='trip_type', hue='booking_complete', data=df, ax=axes[0,1])
sns.countplot(x='wants_extra_baggage', hue='booking_complete', data=df, ax=axes[1,0])
sns.countplot(x='wants_preferred_seat', hue='booking_complete', data=df, ax=axes[1,1])
plt.tight_layout()
plt.show()

# Visualize Numerical Features
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
sns.histplot(df['purchase_lead'], bins=50, kde=True, ax=axes[0,0])
sns.histplot(df['length_of_stay'], bins=50, kde=True, ax=axes[0,1])
sns.histplot(df['flight_duration'], bins=50, kde=True, ax=axes[1,0])
sns.histplot(df['flight_hour'], bins=24, kde=True, ax=axes[1,1])
plt.tight_layout()
plt.show()


# **Step 8: Feature Engineering**

# In[ ]:


# 1. Encoding Categorical Variables
# We will use One-Hot Encoding for low cardinality columns
# and Frequency Encoding for high cardinality columns.

# Check cardinality
for col in categorical_cols:
    print(f"{col}: {df[col].nunique()} unique values")

# One-Hot Encoding for sales_channel, trip_type, flight_day
df = pd.get_dummies(df, columns=['sales_channel', 'trip_type', 'flight_day'], drop_first=True)

# Frequency Encoding for route and booking_origin (due to high unique values)
route_freq = df['route'].value_counts() / len(df)
df['route_freq'] = df['route'].map(route_freq)
df.drop('route', axis=1, inplace=True)

origin_freq = df['booking_origin'].value_counts() / len(df)
df['origin_freq'] = df['booking_origin'].map(origin_freq)
df.drop('booking_origin', axis=1, inplace=True)

# 2. Creating New Features
# Example: Total extra wants
df['total_wants'] = df['wants_extra_baggage'] + df['wants_preferred_seat'] + df['wants_in_flight_meals']

# Example: Ratio of purchase lead to flight duration (avoiding div by zero)
df['lead_duration_ratio'] = df['purchase_lead'] / (df['flight_duration'] + 1)

# Check results
print("\n--- Data after Feature Engineering ---")
print(df.head())
print(df.info())


# **Step 9: Modeling**

# In[ ]:


# ============================================================
# IMPORT LIBRARIES
# ============================================================
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score,
                             precision_score, recall_score, f1_score)
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for high-quality visuals
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# ============================================================
# 1. Separate Features and Target
# ============================================================
X = df.drop('booking_complete', axis=1)
y = df['booking_complete']

# ============================================================
# 2. Split Data (Train & Test)
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 3. Scaling (Optional for Random Forest, but good for consistency)
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 4. Initialize Random Forest Model
# ============================================================
rf_model = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced'
)

# ============================================================
# 5. Cross-Validation
# ============================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv, scoring='roc_auc')

print(f"\n--- Cross-Validation ROC-AUC Scores ---")
print(f"Scores per Fold: {cv_scores}")
print(f"Mean ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================
# 6. Train Model on Training Data
# ============================================================
rf_model.fit(X_train_scaled, y_train)

# ============================================================
# 7. Predict on Test Data
# ============================================================
y_pred = rf_model.predict(X_test_scaled)
y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# ============================================================
# 8. Model Evaluation (Print Metrics)
# ============================================================
print("\n--- Model Evaluation on Test Data ---")
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC   : {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Booked', 'Booked']))

# ============================================================
# 9. Feature Importance
# ============================================================
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n--- Feature Importance (Top 10) ---")
print(feature_importances.head(10))

# ============================================================
# 💾 SAVE VISUALIZATIONS (ADDED SECTION FOR SAVING IMAGES)
# ============================================================

# ------------------------------------------------------------
# FIGURE 1: CONFUSION MATRIX
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Booked', 'Booked'],
            yticklabels=['Not Booked', 'Booked'],
            ax=ax)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: confusion_matrix.png")

# ------------------------------------------------------------
# FIGURE 2: ROC CURVE
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('outputs/roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: roc_curve.png")

# ------------------------------------------------------------
# FIGURE 3: FEATURE IMPORTANCE (TOP 10)
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(10), palette='viridis')
plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: feature_importance.png")

# ------------------------------------------------------------
# FIGURE 4: CROSS-VALIDATION SCORES
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
folds = [f'Fold {i+1}' for i in range(len(cv_scores))]
bars = ax.bar(folds, cv_scores, color='steelblue', edgecolor='black')
ax.axhline(y=cv_scores.mean(), color='red', linestyle='--',
           label=f'Mean = {cv_scores.mean():.4f}')
ax.set_ylim(0.7, 1.0)
ax.set_title('Cross-Validation ROC-AUC Scores (5-Fold)', fontsize=14, fontweight='bold')
ax.set_ylabel('ROC-AUC Score', fontsize=12)
ax.legend()
for bar, score in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{score:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('outputs/cv_scores.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: cv_scores.png")

# ------------------------------------------------------------
# FIGURE 5 (OPTIONAL): ALL FEATURE IMPORTANCES
# ------------------------------------------------------------
plt.figure(figsize=(10, 12))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='coolwarm')
plt.title('All Feature Importances', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('outputs/feature_importance_all.png', dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: feature_importance_all.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*55)
print("📁 ALL VISUALIZATIONS SAVED SUCCESSFULLY!")
print("="*55)
print("""
Files saved in the current directory:
  1. confusion_matrix.png        - Confusion Matrix
  2. roc_curve.png               - ROC Curve (AUC visualization)
  3. feature_importance.png      - Top 10 Features
  4. cv_scores.png               - Cross-Validation Scores
  5. feature_importance_all.png  - All Features (optional)
""")

