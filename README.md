# Car Price Prediction FastAPI

## Description

A service for predicting car prices using simple regression ML models. The project involved comprehensive data preparation, model selection, and optimization steps. Special emphasis was placed on data preprocessing, feature engineering, hyperparameter tuning, and custom metrics evaluation. It also includes a FastAPI service for real-time predictions.

*File contents are currently in Russian, but is going to be fully translated to English.*

### Access

The service has not been deployed but can be run locally.

## Demo

Single prediction (`predict_item`):

![image](https://github.com/user-attachments/assets/77c5d02b-6576-43f4-9598-79a97f9136cd)
![image](https://github.com/user-attachments/assets/09c491a0-21b2-4d62-8df3-9d4b6b7db089)

Working with files (`predict_items`):

![image](https://github.com/user-attachments/assets/e9c570fd-d09c-4828-b274-1f2b812d3785)
![image](https://github.com/user-attachments/assets/6f64fde6-03a9-473f-8a42-58953fd80af0)
![image](https://github.com/user-attachments/assets/c8df9c99-48d6-47d6-958e-cd75a3b14fc4)
![image](https://github.com/user-attachments/assets/c6285115-78c6-4db7-9d8e-85714f42e74d)

## Completed Tasks

1. **Exploratory Data Analysis (EDA):**
   - Analyzed key relationships between features and the target variable.
   - Built correlation heatmaps and identified features with the strongest impact.
   - Discovered potentially significant nonlinear relationships.

2. **Data Preprocessing:**
   - Filled missing values using medians and introduced dummy columns to flag missing data.
   - Applied logarithmic and other transformations to approximate normal distributions.
   - Implemented OneHot encoding for categorical variables while properly handling multicollinearity.

3. **Feature Engineering:**
   - Created new features based on identified relationships, including quadratic and logarithmic transformations.
   - Added additional derived features such as "horsepower per engine volume."

4. **Models:**
   - Built and trained various regression models, including Lasso, Ridge, and ElasticNet.
   - Performed hyperparameter optimization using GridSearchCV with 10-fold cross-validation.
   - Developed a FastAPI service for real-time interaction:
     - Single item price prediction.
     - Bulk price predictions via uploaded CSV files.

5. **Custom Metrics:**
   - Developed business-oriented metrics to evaluate model quality:
     - First metric measured the share of predictions within ±10% of actual values.
     - Second metric penalized under-predictions more heavily, aligning with business priorities.
   - Conducted comparative analysis of models based on these metrics.

## Results

1. **Model Quality:**
   - The `Lasso` model yielded the best results after data preprocessing and feature engineering:
     - $R^2$: `0.8232`
     - $MSE$: `101617170814.93`
   - Custom business metrics confirmed that `Lasso` best met business requirements by minimizing underestimation errors.

2. **Impact of Feature Engineering:**
   - Quadratic transformation of the `year` feature provided the greatest performance boost:
     - Increased $R^2$ from `0.7563` to `0.7931`.
   - Adding the derived `engine`-based feature further improved $R^2$ by an additional `0.03`.

3. **FastAPI Service:**
   - Successfully developed and tested, supporting both single and batch predictions.
   - Provides seamless integration of the predictive model into real-world applications.

## Main Performance Drivers

1. **Feature Creation:**
   - The quadratic transformation of `year` was crucial due to the originally nonlinear relationship with the target.
   - Logarithmic transformations of numeric features (e.g., `max_torque_rpm`) also improved model performance.

2. **Lasso Regression Application:**
   - Regularization eliminated noisy features, enhancing the model's generalization capability.

3. **Custom Business Metrics:**
   - Optimization for custom metrics better aligned model predictions with business requirements.

## Challenges Faced

1. **Removing Irrelevant Features:**
   - Attempts to remove low-importance features decreased performance since many features provided value only when combined with others.

2. **ElasticNet Usage:**
   - ElasticNet did not yield significant improvements, possibly due to the dominance of L1-regularization already utilized by Lasso.

3. **Outlier Handling:**
   - Despite careful analysis, removing outliers did not improve model performance, as these outliers often contained valuable information.

## Conclusion

The project demonstrated the critical importance of thorough data preprocessing and feature experimentation in building effective models. The FastAPI service facilitates easy integration into real business scenarios, making the solution practically applicable. However, further investigation is necessary into effective noise removal and outlier management strategies.
