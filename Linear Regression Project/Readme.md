# Linear Regression Project


![pairplot linear regression project](https://github.com/AdityaDabrase/DSPortfolioProjects/blob/main/DS-ML/Linear%20Regression%20Project/Screenshot%202023-12-08%20135931.png)

## Overview

This project is centered around analyzing customer data to understand the factors influencing yearly spending. We employed linear regression to explore the relationships between various customer attributes and yearly spending.

## Dataset Description

The dataset includes customer information such as email, address, avatar, and numerical columns:

- **Avg. Session Length:** Average session of in-store style advice sessions.
- **Time on App:** Average time spent on the mobile app in minutes.
- **Time on Website:** Average time spent on the website in minutes.
- **Length of Membership:** Number of years the customer has been a member.
- **Yearly Amount Spent:** Total yearly spend by the customer.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Exploratory Data Analysis

We performed exploratory data analysis (EDA) to better understand the dataset. Some of the visualizations include:

- Jointplot of Time on Website vs. Yearly Amount Spent
- Jointplot of Time on App vs. Yearly Amount Spent
- 2D hex bin plot comparing Time on App and Length of Membership
- Pairplot to visualize relationships across all numerical features

## Linear Regression Model

We constructed a linear regression model to predict yearly spending based on customer attributes. Here are the key steps:

1. **Data Splitting:** Splitting the data into training and testing sets.
2. **Model Training:** Training the linear regression model on the training data.
3. **Model Evaluation:** Evaluating the model's performance on the test set.

## Results

### Model Coefficients

The coefficients obtained from the linear regression model are as follows:

- **Avg. Session Length:** 25.98
- **Time on App:** 38.59
- **Time on Website:** 0.19
- **Length of Membership:** 61.28

### Interpretation of Coefficients

Interpreting the coefficients:

- A 1 unit increase in Avg. Session Length is associated with an increase of $25.98 in total dollars spent.
- A 1 unit increase in Time on App is associated with an increase of $38.59 in total dollars spent.
- A 1 unit increase in Time on Website is associated with an increase of $0.19 in total dollars spent.
- A 1 unit increase in Length of Membership is associated with an increase of $61.28 in total dollars spent.

## Conclusion

The analysis suggests that Length of Membership has the strongest correlation with yearly spending, followed by Time on App and Avg. Session Length. The company may consider further exploration into the relationship between Length of Membership and the mobile app or website to make more informed decisions regarding development efforts.

