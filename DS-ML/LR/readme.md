# Logistic Regression Project

![Analyzing 911 Call Data](https://github.com/AdityaDabrase/DSPortfolioProjects/blob/main/DS-ML/LR/Visualization-of-logistic-regression-model-of-an-exemplary-subject-Both-variables.png)

In this project, we aimed to create a predictive model using logistic regression to determine whether an internet user would click on an advertisement based on various user features.

## Dataset Description

The dataset contains the following features:

- **Daily Time Spent on Site:** Consumer time spent on site in minutes
- **Age:** Customer age in years
- **Area Income:** Average income of the geographical area of the consumer
- **Daily Internet Usage:** Average minutes a day consumer is on the internet
- **Ad Topic Line:** Headline of the advertisement
- **City:** City of the consumer
- **Male:** Whether or not the consumer was male
- **Country:** Country of the consumer
- **Timestamp:** Time at which the consumer clicked on the ad or closed the window
- **Clicked on Ad:** Binary value (0 or 1) indicating clicking on the ad

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Exploratory Data Analysis

We conducted exploratory data analysis using Seaborn to understand the dataset better. Some of the visualizations include:

- Histogram of Age distribution
- Jointplot of Area Income vs. Age
- Jointplot of Daily Time spent on site vs. Age with KDE distributions
- Jointplot of Daily Time Spent on Site vs. Daily Internet Usage
- Pairplot showing relationships with the 'Clicked on Ad' feature as hue

## Logistic Regression

We used logistic regression to build a predictive model. Here are the steps:

1. **Data Splitting:** Train-test split with a 33% test size
2. **Model Training:** Logistic regression model fitted on the training set
3. **Model Evaluation:** Classification report generated for model evaluation

## Results

The logistic regression model achieved the following results on the test set:

- Precision: 0.91
- Recall: 0.91
- F1-score: 0.91

## Conclusion

The logistic regression model performed well in predicting whether a user would click on an ad based on the provided features, achieving an average precision, recall, and F1-score of 0.91.

Feel free to explore the Jupyter Notebook for detailed code and analysis.

