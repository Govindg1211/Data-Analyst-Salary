# Data-Analyst-Salary

---

## Introduction to Data Set
This dataset offers insights into Finance & Accounting courses on Udemy, detailing pricing,
ratings, subscriber counts, and course content. With 13,608 courses, it includes both free and
paid options, along with discounts, reviews, and course structure (lectures and practice tests).
The dataset helps analyze online learning trends, identify popular courses, assess pricing
strategies, and evaluate course quality based on user feedback. It is useful for
recommendation systems, trend analysis, pricing optimization, and competitive research in 

---

## Problem Statement: Predicting Data Analyst Salary
The rapid growth of data analytics has led to a surge in demand for data analysts across
various industries. However, salaries for these roles vary widely due to factors like company
size, industry, location, and reputation. Job seekers often struggle to understand salary
determinants, while employers face difficulties in setting competitive pay scales to attract top
talent.
This project focuses on analyzing, modeling, and predicting salary estimates for data analyst
positions using machine learning. By leveraging historical job data, we aim to uncover key
trends and insights that will help both candidates and employers make informed, data-driven
decisions.

---

## The primary challenges this project addresses include:

• Lack of salary transparency – Many job listings do not disclose salary details,
making it difficult for professionals to assess their market value.

• Salary variability – Compensation fluctuates based on factors like industry, company
reputation, location, and ownership structure, complicating salary estimation.

• Influence of company attributes – Examining how company size, revenue, and
employer ratings impact salary levels.


• Effect of job attributes – Investigating whether job titles, application processes, and
industry sectors contribute to salary differences.

• Developing an accurate salary prediction model – Identifying the most effective
machine learning approach for predicting salaries with high precision.


---

## Objective:
The main objectives of this project are:
1. Data Collection & Understanding
   
• Analyze job posting datasets to gain insights into salary distributions and the factors
influencing them.

• Identify the most significant job-related and company-related attributes that impact
salary levels, such as location, industry, company size, and job title.

2. Data Cleaning & Preprocessing

• Handle missing values and inconsistencies in the dataset to ensure data quality.

• Convert categorical variables into numerical features using encoding techniques like
one-hot encoding, target encoding, and frequency encoding.

• Detect and address salary outliers and anomalies in company ratings to maintain
reliable predictions.

3. Exploratory Data Analysis (EDA)

• Visualize salary variations across job titles, industries, and geographic locations.

• Analyze the correlation between salaries and company characteristics, including
revenue, employee count, and employer ratings.

• Investigate whether job postings with "Easy Apply" options have an impact on salary
levels.

4. Feature Engineering

• Implement different encoding strategies to transform categorical variables into
meaningful numerical representations.

• Use statistical techniques and feature selection methods to identify the most influential
features for salary estimation.

• Generate new features, such as experience levels or company prestige scores, to
improve model performance.

5. Predictive Modeling
• Train multiple regression models

• Optimize models using hyperparameter tuning to achieve the highest accuracy.

6. Model Evaluation & Performance Metrics
   

• Use the following metrics to assess model performance:

 ❖ R² Score – To measure how well the model explains salary variance.
 
 ❖ Mean Absolute Error (MAE) – To evaluate average prediction error.
 
 ❖ Mean Squared Error (MSE) – To penalize large prediction errors.

• Compare different models and determine which one provides the best salary prediction
accuracy.

7. Insights & Recommendations
   
• Determine which industries, locations, and company sizes offer the highest salaries.

• Provide job seekers with key insights on the factors that significantly impact salary
offers.

• Offer guidance to companies on optimizing their salary structures to attract top data
analysts.

• Develop an interactive salary prediction tool to help professionals estimate their
expected compensation based on specific job attributes.

• Present findings through visual reports and dashboards for better decision-making in
salary negotiations and job market analysis.

---

### Key Features in the Dataset
This dataset includes a variety of job-related and company-related attributes that play a
crucial role in salary prediction:
1. Job-Related Features: Job Title, Salary Estimate, Job Description, Easy Apply

2. Company-Related Features: Company Name, Company Rating, Headquarters, Company Size,
Founded Year, Type of Ownership

3. Industry & Location-Based Features: Industry, Sector, Location

4. Financial Features: Revenue, Competitors

---

## Techniques considered in the Notebook

1. Data Preprocessing

• Libraries Used: pandas, numpy

• Techniques:
  o Feature selection by dropping unnecessary columns (df.drop(columns=['Avg
Salary']))
  o Splitting data into training and testing sets (train_test_split())

2. Feature Engineering
   
• Handling Multicollinearity:

    o Calculated feature correlation (X_train.corr())
    o Removed features with correlation >0.85 to reduce redundancy
• Recursive Feature Elimination (RFE):

    o Applied RFECV (Recursive Feature Elimination with Cross-Validation) to
    identify the optimal set of features
    o rfecv.fit(X_train, y_train) automatically selects the most important features

3. Machine Learning Models Implemented

• Ensemble Learning Models:

      o RandomForestRegressor()
      o GradientBoostingRegressor()
      o XGBRegressor()
      o LGBMRegressor()
      
• Linear Model for Feature Selection:

    o LinearRegression() used in RFECV
    
4. Hyperparameter Tuning
   
• Optimized parameter dictionaries (rf_params, xgb_params, gb_params, lgbm_params)
for:

    o Number of estimators
    o Maximum depth
    o Learning rate
    o Subsampling strategies
    o Regularization parameters (lambda_l1, lambda_l2)

5. Model Evaluation
   
• Performance Metrics:

    o R² Score (train, test, and cross-validation)
    o Root Mean Squared Error (RMSE)
    o Mean Absolute Error (MAE)

• Cross-Validation:

    o Applied KFold cross-validation (cv = KFold(n_splits=5))
    o Used cross_val_score() to assess model performance on unseen data
    
• Model Comparison:

    o Visualized Test R² and CV R² using a bar plot (seaborn)

6. Model Saving

• The best-performing model (XGBRegressor) was saved using joblib.dump(xgb_model,
"best_gb_model.pkl")

---

## Step-by-Step Project Implementation

Step 1: Import Required Libraries

• Various Python libraries are used for data handling (pandas, numpy), data visualization
(seaborn, matplotlib), and machine learning (scikit-learn, xgboost, lightgbm).

• joblib is utilized to save the trained model for future use.


Step 2: Load and Prepare the Dataset

• The dataset is loaded into a Pandas DataFrame.

• Features (X) and the target variable (y), which represents the average salary, are
separated for model training.

• This step ensures the data is properly structured before further processing.


Step 3: Split Data into Training and Testing Sets

• The dataset is divided into:

    o Training Set (80%) – Used for training models.
    o Testing Set (20%) – Used to evaluate model performance on unseen data.
    
• A random_state value is set to ensure reproducibility.


Step 4: Remove Highly Correlated Features

• A correlation matrix is generated to examine relationships between numerical features.

• Features with a correlation above 0.85 are removed to prevent multicollinearity.

• Eliminating redundant features ensures the model learns from unique and relevant
information.


Step 5: Feature Selection Using Recursive Feature Elimination (RFE)

• Recursive Feature Elimination with Cross-Validation (RFECV) is applied to select the
most important features.

• A Linear Regression model serves as the base estimator for feature selection.

• Only the most relevant features are retained, improving model efficiency and
predictive accuracy.


Step 6: Define Hyperparameters for Models

• Hyperparameters control how a model learns from data.

• Key parameters are tuned for Random Forest, XGBoost, Gradient Boosting, and
LightGBM, including:

    o Number of trees (n_estimators)
    o Tree depth (max_depth)
    o Learning rate (step size for updates)
    o Subsampling and regularization techniques to prevent overfitting


Step 7: Initialize Machine Learning Models

• Four different models are selected:
    
    o Random Forest Regressor: Uses multiple decision trees for prediction.
    o XGBoost Regressor: A high-performance gradient boosting algorithm.
    o Gradient Boosting Regressor: Similar to XGBoost but often slower.
    o LightGBM Regressor: An optimized gradient boosting model for large datasets.

• Each model is initialized with pre-tuned hyperparameters.


Step 8: Define Model Evaluation Metrics

• A function is created to assess model performance using multiple metrics:

    o R² Score – Measures how well the model explains the variance in the target
    variable.

    o Root Mean Squared Error (RMSE) – Evaluates average prediction error.

    o Mean Absolute Error (MAE) – Measures the absolute difference between
    predicted and actual values.

• A 5-fold cross-validation technique is used to test model consistency across different
data subsets.


Step 9: Train and Evaluate Models

• Each model is trained using the training data.

• Predictions are generated for both training and testing datasets.

• Performance is assessed using R² Score, RMSE, and MAE.

• Cross-validation scores are computed to evaluate model stability.


Step 10: Compare Model Performance

• A bar chart is used to compare the performance of all four models.

• Two key scores are plotted:

    o Test R² Score: Evaluates model accuracy on unseen data.

    o Cross-validation R² Score: Measures performance across different training
    subsets.

• This comparison helps identify the best-performing model.


Step 11: Save the Best Model

• Based on evaluation metrics, XGBoost Regressor is selected as the top-performing
model.

• The trained model is saved using joblib.dump(), allowing for future use without
retraining.

---

## Summary
This project focuses on predicting average salaries using machine learning models, including
Random Forest, XGBoost, Gradient Boosting, and LightGBM. The dataset underwent
preprocessing, where highly correlated features were removed to prevent multicollinearity,
and Recursive Feature Elimination (RFE) was applied for optimal feature selection. The
models were trained on the processed dataset and evaluated using R² Score, RMSE, MAE,
and cross-validation to ensure accuracy and reliability. Among all models, XGBoost
outperformed the others, demonstrating high accuracy and stability in salary prediction.
The final trained model was saved using joblib.dump() for future use. This study highlights
how machine learning can be leveraged to estimate salaries based on key job-related features.
Future improvements could involve integrating additional job-related attributes, such as
industry trends, company benefits, or job market demand, to enhance prediction accuracy.
Additionally, deploying this model in a real-world application or as a web-based tool could
provide users with instant salary estimates based on their skills and experience.

---

## Conclusion
Insights and Future Directions for Salary Prediction Models
The key findings are:

• Feature selection and correlation analysis played a crucial role in improving model
accuracy by removing redundant and irrelevant features.

• XGBoost outperformed other models, excelling in handling complex patterns and
delivering stable predictions.

• The trained model has real-world applicability, making it useful for salary estimation
across various job roles.


In future work, further improvements can be made by:

• Enhancing Feature Set – Adding factors such as industry trends, job experience, and
location-based salary variations to refine predictions.

• xploring Deep Learning – Investigating advanced neural networks for potentially
higher accuracy and better generalization

• Model Deployment – Developing a web-based tool for real-time salary prediction,
allowing users to estimate salaries based on their profile.

By incorporating these improvements, the model can evolve into a powerful tool for salary
estimation and career planning

---
