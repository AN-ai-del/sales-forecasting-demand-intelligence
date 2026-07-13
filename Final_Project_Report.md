# Sales Forecasting and Demand Intelligence
## Final Project Report

**Author:** Anushka Das

**Project Type:** End-to-End Machine Learning Project

**Domain:** Retail Sales Forecasting

---

# Executive Summary

Accurate sales forecasting enables organizations to optimize inventory planning, workforce allocation, budgeting, and supply chain operations. This project develops a complete machine learning pipeline that forecasts future retail sales using historical transaction data and engineered time-based features.

The solution includes data preprocessing, exploratory data analysis, feature engineering, model training, explainability, performance comparison, and deployment through an interactive Streamlit dashboard. Three regression algorithms—Linear Regression, Random Forest, and XGBoost—were evaluated, with Linear Regression selected as the final production model based on objective evaluation metrics.

---

# Business Problem

Retail organizations often experience fluctuating customer demand. Without reliable sales forecasts, businesses may face:

- Overstocking inventory
- Product shortages
- Revenue loss
- Inefficient workforce planning
- Poor budgeting decisions

The objective of this project is to build a machine learning solution capable of forecasting future sales using historical retail data.

---

# Project Objectives

The project was designed to:

- Analyze historical sales behaviour
- Engineer forecasting features
- Train multiple regression models
- Compare predictive performance
- Interpret model behaviour
- Generate future sales predictions
- Build an interactive business dashboard

---

# Dataset Overview

The project uses historical retail sales data containing daily sales transactions.

After preprocessing, the forecasting dataset contains:

- Daily sales observations
- Calendar-based features
- Lag features
- Rolling average features

Target Variable:

- Sales

Engineered Features:

- Year
- Month
- Quarter
- Week
- Day
- DayOfWeek
- DayOfYear
- IsWeekend
- Lag_1
- Lag_7
- Lag_30
- Rolling_Mean_7
- Rolling_Mean_30

---

# Data Preparation

The preprocessing pipeline included:

- Missing-value validation
- Date conversion
- Daily sales aggregation
- Feature engineering
- Numeric validation
- Dataset cleaning
- Feature consistency checks

The processed dataset was stored separately from the raw data to maintain reproducibility.

---

# Exploratory Data Analysis

Exploratory analysis identified several important business trends.

Key analyses included:

- Daily sales trends
- Monthly sales trends
- Yearly sales comparison
- Quarterly sales behaviour
- Weekday analysis
- Sales distribution
- Seasonal patterns

These visualizations provide insights into demand fluctuations and long-term sales behaviour.

---

# Feature Engineering

Time-series forecasting performance depends heavily on meaningful temporal features.

The following feature groups were engineered:

## Calendar Features

- Year
- Month
- Quarter
- Week
- Day
- DayOfWeek
- DayOfYear
- IsWeekend

## Lag Features

- Previous observation (Lag_1)
- Seven observations earlier (Lag_7)
- Thirty observations earlier (Lag_30)

## Rolling Features

- Rolling Mean (7)
- Rolling Mean (30)

These features allow the models to capture seasonality, trends, and recent sales behaviour.

---

# Machine Learning Models

Three regression algorithms were trained.

## Linear Regression

A simple and interpretable regression model that served as the final production model.

## Random Forest

An ensemble learning algorithm capable of modelling nonlinear relationships.

## XGBoost

A gradient boosting algorithm designed for high predictive performance.

Each model was trained using identical feature inputs to ensure fair comparison.

---

# Model Evaluation

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Performance Summary

| Model | MAE | RMSE | R² |
|------|------:|------:|------:|
| Linear Regression | 1358.46 | 2313.59 | 0.2392 |
| Random Forest | 1367.22 | 2451.95 | 0.1455 |
| XGBoost | 1405.12 | 2525.25 | 0.0937 |

Linear Regression achieved the best overall performance and was selected as the final forecasting model.

---

# Model Explainability

Model transparency was incorporated through explainability analysis.

The dashboard provides:

- Linear Regression coefficient analysis
- Random Forest feature importance
- XGBoost feature importance
- Cross-model feature comparison
- Business interpretation of important variables

This enables users to understand why predictions are generated rather than treating the models as black boxes.

---

# Dashboard Features

The Streamlit dashboard includes six interactive pages:

- Executive Dashboard
- Dataset Overview
- Sales Trend Analysis
- Interactive Forecast
- Model Comparison
- Model Explainability

Key capabilities include:

- Interactive forecasting
- Automatic calendar feature generation
- Reliability assessment
- Prediction history
- Interactive visualizations
- Downloadable datasets
- Business insights

---

# Business Value

This forecasting system can support organizations by:

- Improving inventory planning
- Reducing stock shortages
- Supporting production planning
- Improving budgeting decisions
- Assisting workforce scheduling
- Identifying seasonal demand patterns
- Supporting data-driven business decisions

---

# Limitations

Current limitations include:

- Limited historical observations
- No external economic indicators
- No promotional data
- No holiday information
- Single-step forecasting
- Moderate predictive accuracy due to limited features

---

# Future Improvements

Potential future enhancements include:

- Holiday and promotional features
- Weather integration
- Price elasticity analysis
- Inventory-aware forecasting
- Multi-step forecasting
- Hyperparameter optimization
- SHAP explainability
- Automated model retraining
- API deployment
- Docker containerization
- Streamlit Cloud deployment
- CI/CD integration

---

# Technologies Used

Programming Language

- Python

Libraries

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Plotly
- Streamlit
- Matplotlib

Tools

- VS Code
- Git
- GitHub

---

# Conclusion

This project demonstrates an end-to-end machine learning workflow for retail sales forecasting. Beginning with raw sales data, the pipeline performs preprocessing, feature engineering, model development, evaluation, explainability, and interactive deployment through Streamlit.

Among the evaluated algorithms, Linear Regression achieved the best predictive performance and was selected as the production model. The resulting dashboard enables business users to analyze historical trends, compare models, generate forecasts, and interpret model behaviour in an intuitive interface.

The project showcases practical skills in data preprocessing, time-series feature engineering, machine learning, visualization, model interpretation, and application deployment, making it suitable as both a portfolio project and a foundation for production-ready forecasting systems.