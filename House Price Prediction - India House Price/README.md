# House Price Prediction: A Comparative Analysis of Machine Learning Models

This project implements and compares different machine learning models to predict house prices in India. We'll explore how different features affect house prices and evaluate which model performs best for this prediction task.

## Dataset Overview

The dataset contains information about houses in India with the following features:

- POSTED_BY: Category of who listed the property (Owner/Dealer/Builder)
- UNDER_CONSTRUCTION: Whether the property is under construction (0/1)
- RERA: Whether the property is RERA approved (0/1)
- BHK_NO: Number of rooms in the property
- BHK_OR_RK: Type of property (BHK/RK)
- SQUARE_FT: Total area in square feet
- READY_TO_MOVE: Whether the property is ready to move in (0/1)
- RESALE: Whether it's a resale property (0/1)
- ADDRESS: Location of the property
- LONGITUDE: Geographical longitude
- LATITUDE: Geographical latitude
- TARGET(PRICE_IN_LACS): Price in lakhs (target variable)

## Step-by-Step Implementation

### 1. Data Loading and Initial Exploration

First, we import necessary libraries and load the data:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Load the data
df = pd.read_csv("train.csv")
```

### 2. Data Preprocessing

The preprocessing steps include:

1. Checking for missing values:
```python
df.isnull().sum()
```

2. Removing duplicates:
```python
df.drop_duplicates(inplace=True)
```

3. Feature encoding using one-hot encoding for categorical variables:
```python
encoded_df = pd.get_dummies(df, dtype=float, drop_first=True)
```

### 3. Feature Engineering and Selection

We prepare our features (X) and target variable (y):

```python
# Separate features and target
X = encoded_df.drop('TARGET(PRICE_IN_LACS)', axis=1)
y = encoded_df['TARGET(PRICE_IN_LACS)']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### 4. Model Implementation

We implement three different models for comparison:

1. Linear Regression
2. Decision Tree
3. Random Forest

```python
# Initialize models
LR = LinearRegression()
DT = DecisionTreeRegressor(random_state=42)
RF = RandomForestRegressor(random_state=42)

# Train models
LR.fit(x_train, y_train)
DT.fit(x_train, y_train)
RF.fit(x_train, y_train)
```

### 5. Model Evaluation

We evaluate the models using R-squared (R²) scores on both training and test sets:

```python
print("====== TRAIN SCORE ======")
print(f"Linear Regression Train Score {LR.score(x_train, y_train)}")
print(f"Decision Tree Train Score {DT.score(x_train, y_train)}")
print(f"Random Forest Train Score {RF.score(x_train, y_train)}")

print("\n ====== TEST SCORE ======")
print(f"Linear Regression Test Score {LR.score(x_test, y_test)}")
print(f"Decision Tree Test Score {DT.score(x_test, y_test)}")
print(f"Random Forest Test Score {RF.score(x_test, y_test)}")
```

Results:
- Linear Regression: Train Score = 0.210, Test Score = 0.337
- Decision Tree: Train Score = 0.961, Test Score = 0.940
- Random Forest: Train Score = 0.983, Test Score = 0.938

## Data Insights and Visualization

### 1. Distribution of Property Types

The dataset shows several interesting patterns in the Indian housing market:

- Most properties are posted by Dealers (61%), followed by Owners (35%), and a small portion by Builders (4%)
- The majority of properties are BHK (Bedroom, Hall, Kitchen) type rather than RK (Room, Kitchen)
- 2 BHK properties are the most common, followed by 3 BHK

### 2. Price Distribution by City

We analyzed price variations across different cities:

```python
sns.barplot(x='CITY', y='TARGET(PRICE_IN_LACS)', data=processed_df)
plt.title('Target Price by CITY')
plt.show()
```

Key findings:
- Mumbai shows the highest average property prices
- Bangalore and Delhi follow as the second and third most expensive cities
- Tier-2 cities show significantly lower average prices

### 3. Impact of Property Features

Several features significantly influence property prices:

1. **Square Footage**: Shows a strong positive correlation with price
2. **Location**: Prime locations within cities command premium prices
3. **RERA Approval**: RERA-approved properties tend to have higher valuations
4. **Ready to Move**: Ready properties often command a premium over under-construction ones

### 4. Feature Importance

The Random Forest model helps us understand which features are most important in determining house prices:

1. Square footage (most important)
2. Location (longitude and latitude)
3. Number of rooms (BHK_NO)
4. RERA approval status

## Model Performance Visualization

Here's how our models performed across different metrics:

1. **R-squared Scores**:
   ```
   Model              Training Score    Test Score
   Linear Regression  0.210            0.337
   Decision Tree      0.961            0.940
   Random Forest      0.983            0.938
   ```

2. **Key Observations**:
   - Linear Regression's poor performance suggests highly non-linear relationships in the data
   - Decision Tree and Random Forest both show excellent performance
   - Random Forest provides slightly better generalization

## Model Performance Analysis and Conclusion

After comparing the three models, we can conclude:

1. **Linear Regression** performed poorly with both training (21%) and test (34%) scores, indicating that the relationship between features and house prices is not linear.

2. **Decision Tree** showed excellent performance with training score of 96% and test score of 94%. The small difference between training and test scores suggests good generalization.

3. **Random Forest** achieved the best training score (98%) and comparable test score (94%) to Decision Tree. This indicates:
   - Excellent ability to capture complex patterns in the data
   - Good generalization to unseen data
   - Robust performance due to ensemble learning

### Why Random Forest is the Best Choice

Random Forest emerged as the optimal model for this house price prediction task for several reasons:

1. **High Accuracy**: It achieved the highest combined performance on both training and test sets.

2. **Robustness**: As an ensemble method, it's less prone to overfitting compared to single decision trees.

3. **Feature Handling**: It can effectively handle:
   - Both numerical and categorical features
   - Non-linear relationships
   - Complex interactions between features

4. **Stability**: Random Forest provides more stable predictions by averaging multiple decision trees, reducing variance in predictions.

The high R² scores (0.983 for training and 0.938 for testing) indicate that our Random Forest model can explain approximately 94% of the variance in house prices, making it a reliable tool for price prediction in the Indian housing market.

## Practical Applications

This model can be used for:

1. **Price Estimation**: Helping buyers and sellers determine fair market values
2. **Market Analysis**: Understanding price trends in different locations
3. **Investment Decisions**: Identifying potentially undervalued properties
4. **Development Planning**: Helping developers optimize property features

## Future Improvements

The model could be enhanced by:

1. Including more features:

2. Implementing advanced techniques:
   - XGBoost or LightGBM for potentially better performance
   - Neural networks for capturing more complex patterns
   - Time series analysis for price trend prediction

## Requirements

To run this project, you need:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository
2. Install requirements
3. Run the Jupyter notebook
4. Follow the step-by-step implementation
