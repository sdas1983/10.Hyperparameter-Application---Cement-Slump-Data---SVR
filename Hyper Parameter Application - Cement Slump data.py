# Hyperparameter Application - Cement Slump Data - SVR

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import uniform  # Correctly importing uniform from scipy.stats

# Set display options
pd.set_option('display.max_columns', None)

# Load the data
df = pd.read_csv(r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\cement_slump.csv")

# Data preprocessing
df.drop('Unnamed: 0', axis=1, inplace=True)
print("Missing Values:\n", df.isnull().sum())
print("Data Shape:", df.shape)
print("Correlation Matrix:\n", df.corr())

# Heatmap of correlations
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Features and target variable
X = df.drop(['Compressive Strength (28-day)(Mpa)'], axis=1)
y = df['Compressive Strength (28-day)(Mpa)']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initial SVR model
base_model = SVR()
base_model.fit(X_train, y_train)

# Predictions and evaluation
base_preds = base_model.predict(X_test)
print("Base Model MAE:", mean_absolute_error(y_test, base_preds))
print("Base Model RMSE:", np.sqrt(mean_squared_error(y_test, base_preds)))
print("Test Set Mean Value:", y_test.mean())

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'C': [0.001, 0.01, 0.1, 0.5, 1],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': [2, 3, 4],
    'epsilon': [0, 0.01, 0.1, 0.5, 1, 2]
}

grid_model = GridSearchCV(SVR(), param_grid)
grid_model.fit(X_train, y_train)

print("Best Parameters from Grid Search:", grid_model.best_params_)

# Predictions and evaluation with best parameters
grid_preds = grid_model.predict(X_test)
print("Grid Model MAE:", mean_absolute_error(y_test, grid_preds))
print("Grid Model RMSE:", np.sqrt(mean_squared_error(y_test, grid_preds)))

# Hyperparameter tuning with RandomizedSearchCV
distributions = dict(C=uniform(loc=2, scale=10))  # Correcting the uniform distribution
random_model = RandomizedSearchCV(SVR(), distributions, random_state=0)
random_model.fit(X_train, y_train)

# Predictions and evaluation with randomized search
print("Randomized Search Model MAE:", mean_absolute_error(y_test, grid_preds))
print("Randomized Search Model RMSE:", np.sqrt(mean_squared_error(y_test, grid_preds)))

# Random search with more complex distributions
rand_list = {
    "C": uniform(2, 10),  # Correcting the uniform distribution
    "gamma": uniform(0.1, 1)  # Correcting the uniform distribution
}

rand_search = RandomizedSearchCV(SVR(), 
                                 param_distributions=rand_list, 
                                 n_iter=20, 
                                 n_jobs=4, 
                                 cv=3, 
                                 random_state=42)
rand_search.fit(X_train, y_train)

# Final evaluation
print("Random Search MAE:", mean_absolute_error(y_test, grid_preds))
print("Random Search RMSE:", np.sqrt(mean_squared_error(y_test, grid_preds)))
