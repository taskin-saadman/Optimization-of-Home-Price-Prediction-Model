# Optimization-of-Home-Price-Prediction-Model
A Python-based implementation of a machine learning pipeline to predict home prices using the Decision Tree Regressor. The model is optimized by identifying the best tree size (maximum leaf nodes) that minimizes the Mean Absolute Error (MAE) on validation data.

The dataset was sourced from Kaggle's _Intro to Machine Learning_ course (https://www.kaggle.com/learn/intro-to-machine-learning)
The code is the solution to one of the exercises in the course.

**Key Features:**
**Dataset**: The analysis uses the home_data.csv dataset, which includes features like lot area, year built, square footage, and the number of rooms.
**Data Preprocessing**: The target variable (SalePrice) and relevant features are extracted for training.

**Model Optimization**:
A range of candidate leaf nodes is evaluated.
The model with the optimal number of leaf nodes is selected based on MAE performance on the validation set.
**Final Model:** The best-performing model is trained on the entire dataset to predict home prices.
**Validation:** The predictions from the optimized model are compared with actual prices for accuracy.
