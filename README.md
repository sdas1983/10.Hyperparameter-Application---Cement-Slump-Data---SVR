# Hyperparameter Tuning for SVR on Cement Slump Data

This project demonstrates the application of Support Vector Regression (SVR) and hyperparameter tuning using Grid Search and Randomized Search on the Cement Slump dataset. The dataset is processed, and models are trained to predict the compressive strength of cement based on various features.

## Dataset

The dataset used in this project contains various features related to the properties of cement, including:

- **Cement Content**
- **Slag Content**
- **Fly Ash Content**
- **Water Content**
- **Superplasticizer Content**
- **Coarse Aggregate**
- **Fine Aggregate**
- **Age**
- **Compressive Strength (28-day)(Mpa)** (Target variable)

## Project Structure

- **Data Loading**: The data is loaded from a CSV file, and unnecessary columns are dropped.
- **Data Preprocessing**: Missing values are checked, and correlations between features are visualized using a heatmap.
- **Feature Engineering**: The features are separated from the target variable, and the data is split into training and testing sets. Feature scaling is applied using `StandardScaler`.
- **Model Training**: 
  - **Base Model**: A base SVR model is trained and evaluated.
  - **Grid Search**: Hyperparameter tuning using Grid Search is applied to find the best parameters for SVR.
  - **Randomized Search**: Another hyperparameter tuning technique, Randomized Search, is applied to explore a broader range of parameters.
- **Model Evaluation**: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) are calculated for model evaluation.

## Requirements

To run this code, you'll need the following Python packages:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `scipy`

You can install the required packages using `pip`:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```
## Project Highlights
- Data Visualization: Correlation heatmap to understand relationships between features.
- SVR Modeling: Application of SVR to predict the compressive strength of cement.
- Hyperparameter Tuning: Detailed hyperparameter tuning using Grid Search and Randomized Search to optimize model performance.
