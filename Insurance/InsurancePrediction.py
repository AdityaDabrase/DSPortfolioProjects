# %% [markdown]
#  # Health Insurance Price prediction

# %% [markdown]
# Many factors that affect how much you pay for health insurance are not within your control. Nonetheless, it's good to have an understanding of what they are. Here are some factors that affect how much health insurance premiums cost
# 
# age: age of primary beneficiary
# 
# sex: insurance contractor gender, female, male
# 
# bmi: Body mass index, providing an understanding of body, weights that are relatively high or low relative to height, objective index of body weight (kg / m ^ 2) using the ratio of height to weight, ideally 18.5 to 24.9
# 
# children: Number of children covered by health insurance / Number of dependents
# 
# smoker: Smoking
# 
# region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# %%
df = pd.read_csv('insurance.csv')
df.head()

# %%
df.shape


# %%
df.describe()


# %%
df.dtypes


# %%
df.isnull().sum()


# %% [markdown]
# We have 0 missing values which is very good. Now let's do EDA with some cool graphs :) First we'll see how the charges are distributed according to given factors

# %%
import seaborn as sns
import matplotlib.pyplot as plt


sns.set(style='darkgrid')
f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.distplot(df['charges'], kde=True, color='steelblue', hist_kws={'edgecolor': 'black'})

# Add a title and adjust labels
plt.title('Distribution of Insurance Charges', fontsize=16)
plt.xlabel('Charges', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()


# %% [markdown]
# This distribution is right-skewed. To make it closer to normal we can apply natural log
# 
# 

# %% [markdown]
# This transformation is commonly applied when dealing with right-skewed distributions. Here's an explanation of why:
# 
# Symmetry: Applying the natural logarithm to a right-skewed distribution can help make the distribution more symmetric. Right-skewed distributions often have a long right tail, and taking the logarithm tends to compress the larger values, pulling in the tail and making the distribution more symmetric.
# 
# Stabilizing Variances: In some cases, taking the logarithm can stabilize the variances across different levels of the independent variable. This can be beneficial in statistical modeling, especially when you're dealing with data where the spread of values increases with the level of the independent variable.
# 
# Interpretability: When interpreting results in the context of linear models, taking the logarithm can make the effects of predictor variables more interpretable. For instance, in a linear regression, the coefficient for a variable after a log transformation represents the percentage change in the dependent variable for a one-unit change in the predictor.
# 
# Normality: Transforming data with the natural logarithm can also make the distribution more normal or close to normal. This can be advantageous in statistical analyses that assume normality, such as certain parametric tests or linear regression.

# %%
f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.distplot(np.log10(df['charges']), kde = True, color = 'seagreen' )

# %% [markdown]
# Now let's look at the charges by region
# 
# 

# %%
charges = df['charges'].groupby(df.region).sum().sort_values(ascending = True)
f, ax = plt.subplots(1, 1, figsize=(8, 6))
ax = sns.barplot(x=charges.values, y=charges.index, palette='Blues')
# Adding labels and title
plt.xlabel('Sum of Charges')
plt.ylabel('Region')
plt.title('Sum of Charges by Region')

# Show the plot
plt.show()

# %%
charges

# %%

sns.set(style='whitegrid')

charges = df['charges'].groupby(df['region']).sum().sort_values(ascending=True)

color_palette = sns.color_palette("Greens", len(charges))

# Create a rounded-corner horizontal bar plot
f, ax = plt.subplots(figsize=(10, 6))
ax = sns.barplot(x=charges.values, y=charges.index, palette=color_palette, edgecolor=".9")

# Adding labels and title with a larger font size
plt.xlabel('Sum of Charges', fontsize=14)
plt.ylabel('Region', fontsize=14)
plt.title('Sum of Charges by Region', fontsize=16)

# Set a background color for the plot
ax.set_facecolor("#f5f5f5")

# Add a shadow to the plot for a raised effect
ax.spines['top'].set_linewidth(0)
ax.spines['right'].set_linewidth(0)
ax.spines['bottom'].set_linewidth(0)
ax.spines['left'].set_linewidth(0)

plt.show()


# %% [markdown]
# So overall the highest medical charges are in the Southeast and the lowest are in the Southwest. Taking into account certain factors (sex, smoking, having children) let's see how it changes by region
# 
# 

# %%
f, ax = plt.subplots(1,1, figsize=(12,8))
ax = sns.barplot(x = 'region', y = 'charges',
                 hue='smoker', data=df, palette='cubehelix')

# %%
f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.barplot(x='region', y='charges', hue='sex', data=df, palette='viridis')

# %%
f, ax = plt.subplots(1, 1, figsize=(12, 8))
ax = sns.barplot(x='region', y='charges', hue='children', data=df, palette='spring')

# %% [markdown]
# Observing the bar plots, it is evident that the Southeast region exhibits the highest medical charges attributable to smoking, while the Northeast region demonstrates the lowest. Notably, individuals in the Southwest display a higher prevalence of smoking compared to those in the Northeast. However, despite higher smoking rates, individuals in the Northeast exhibit greater medical charges by gender when compared to those in the Southwest and Northwest regions collectively. Additionally, there is a discernible trend indicating that individuals with dependents tend to incur elevated medical expenses overall.

# %% [markdown]
# ## Analysing the medical charges by age, bmi and children according to the smoking factor
# 
# 

# %%
ax = sns.lmplot(x = 'age', y = 'charges', data=df, hue='smoker', palette='Set2')
ax = sns.lmplot(x = 'bmi', y = 'charges', data=df, hue='smoker', palette='viridis')
ax = sns.lmplot(x = 'children', y = 'charges', data=df, hue='smoker', palette='Set3')

# %% [markdown]
# Smoking has the highest impact on medical costs, even though the costs are growing with age, bmi and children. Also people who have children generally smoke less, which the following violinplots shows too
# 
# 

# %%
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = sns.violinplot(x = 'children', y = 'charges', data=df,
                 orient='v', hue='smoker', palette='Set3')

# %% [markdown]
# The violin plot created using Seaborn depicts the distribution of medical charges in relation to the number of children an individual has, with a further breakdown based on smoking status. The horizontal axis represents the number of children (from 0 to a higher count), while the vertical axis represents the corresponding medical charges. The width of each violin plot at a given number of children reflects the density of data points, with wider sections indicating higher concentration. The plot is split into different hues, representing smokers and non-smokers, enabling a visual comparison of their respective distributions. The violin plot reveals insights into the variability and central tendency of medical charges across different child counts and smoking categories, providing a comprehensive view of the data distribution and potential relationships between the variables.

# %%
##Converting objects labels into categorical
df[['sex', 'smoker', 'region']] = df[['sex', 'smoker', 'region']].astype('category')
df.dtypes

# %%
##Converting category labels into numerical using LabelEncoder
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
label.fit(df.sex.drop_duplicates())
df.sex = label.transform(df.sex)
label.fit(df.smoker.drop_duplicates())
df.smoker = label.transform(df.smoker)
label.fit(df.region.drop_duplicates())
df.region = label.transform(df.region)
df.dtypes

# %%
f, ax = plt.subplots(1, 1, figsize=(10, 10))
ax = sns.heatmap(df.corr(), annot=True, cmap='RdYlBu')

# %% [markdown]
# # Regression Model Comparison
# 
# ## Linear Regression
# Linear Regression is a straightforward and easily interpretable model, making it suitable for scenarios where the relationship between features and the target is linear. It is known for its fast training and prediction. However, it becomes prone to overfitting in high-dimensional data.
# 
# ## Ridge Regression
# Ridge Regression handles multicollinearity effectively and incorporates a regularization term to mitigate overfitting. It is a suitable choice when dealing with multicollinearity, striking a balance between simplicity and accuracy. Tuning is required for the introduced hyperparameter.
# 
# ## Lasso Regression
# Lasso Regression performs feature selection by setting some coefficients to zero, making it advantageous in scenarios where feature selection is crucial or when dealing with high-dimensional data. However, it may be sensitive to outliers.
# 
# ## Random Forest Regressor
# Random Forest Regressor excels in handling non-linearity and complex relationships. Its robustness to outliers and reduction of overfitting by averaging multiple decision trees make it suitable for datasets with intricate patterns. However, it is less interpretable and can be computationally expensive.
# 
# ## Polynomial Regression
# Polynomial Regression captures non-linear relationships and can model complex patterns. It is useful when the relationship between features and the target is not adequately captured by a simple linear model. However, it is prone to overfitting, especially with high-degree polynomials.
# 

# %% [markdown]
# ## Linear Regression
# 
# 

# %%
from sklearn.model_selection import train_test_split as holdout
from sklearn.linear_model import LinearRegression
from sklearn import metrics
x = df.drop(['charges'], axis = 1)
y = df['charges']
x_train, x_test, y_train, y_test = holdout(x, y, test_size=0.2, random_state=0)
Lin_reg = LinearRegression()
Lin_reg.fit(x_train, y_train)
print(Lin_reg.intercept_)
print(Lin_reg.coef_)
print(Lin_reg.score(x_test, y_test))

# %%

This code implements a simple linear regression model using the scikit-learn library in Python. The dataset is split into training and testing sets, with 80% used for training and 20% for testing. The linear regression model is created and trained on the training set. The code then prints the intercept and coefficients of the model, providing insight into the relationship between the independent variables and the target variable ('charges'). Additionally, the coefficient of determination (R²) is calculated and printed, representing the proportion of variance in the dependent variable that the model explains. The output serves as a summary of the model's parameters and its performance on the test set.

Intercept and Coefficients:

The intercept term indicates the predicted value when all independent variables are zero.
Coefficients represent the change in the dependent variable for a one-unit change in the corresponding independent variable.
R² Score:

The R² score quantifies the model's ability to explain the variability in the target variable.
A higher R² score indicates a better fit between the model and the observed data.

# %% [markdown]
# 
# 
# 
# ## Ridge Regression
# 
# 

# %%
from sklearn.linear_model import Ridge
Ridge = Ridge(alpha=0.5)
Ridge.fit(x_train, y_train)
print(Ridge.intercept_)
print(Ridge.coef_)
print(Ridge.score(x_test, y_test))

# %% [markdown]
# ## Lasso Regression
# 
# 

# %%
from sklearn.linear_model import Lasso
Lasso = Lasso(alpha=0.2, fit_intercept=True, precompute=False, max_iter=1000,
              tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
Lasso.fit(x_train, y_train)
print(Lasso.intercept_)
print(Lasso.coef_)
print(Lasso.score(x_test, y_test))

# %% [markdown]
# ## Random Forest Regressor
# 

# %%


from sklearn.ensemble import RandomForestRegressor as rfr
x = df.drop(['charges'], axis=1)
y = df.charges

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Create and train the RandomForestRegressor
rfr_model = RandomForestRegressor(n_estimators=100, criterion='squared_error', random_state=1)
rfr_model.fit(x_train, y_train)

# Make predictions on the training and test sets
y_train_pred = rfr_model.predict(x_train)
y_test_pred = rfr_model.predict(x_test)

# Evaluate the model
mse_train = metrics.mean_squared_error(y_train, y_train_pred)
mse_test = metrics.mean_squared_error(y_test, y_test_pred)

r2_train = metrics.r2_score(y_train, y_train_pred)
r2_test = metrics.r2_score(y_test, y_test_pred)

# Print the results
print('MSE train data: %.3f, MSE test data: %.3f' % (mse_train, mse_test))
print('R2 train data: %.3f, R2 test data: %.3f' % (r2_train, r2_test))

# %%
plt.figure(figsize=(8,6))

plt.scatter(x_train_pred, x_train_pred - y_train,
          c = 'b', marker = 'o', s = 35, alpha = 0.5,
          label = 'Train data')
plt.scatter(x_test_pred, x_test_pred - y_test,
          c = 'r', marker = 'o', s = 35, alpha = 0.7,
          label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Actual values')
plt.legend(loc = 'upper right')
plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


# %%
print('Feature importance ranking\n\n')
importances = Rfr.feature_importances_
std = np.std([tree.feature_importances_ for tree in Rfr.estimators_],axis=0)
indices = np.argsort(importances)[::-1]
variables = ['age', 'sex', 'bmi', 'children','smoker', 'region']
importance_list = []
for f in range(x.shape[1]):
    variable = variables[indices[f]]
    importance_list.append(variable)
    print("%d.%s(%f)" % (f + 1, variable, importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(importance_list, importances[indices],
       color="b", yerr=std[indices], align="center")

# %% [markdown]
# 
# ## Polynomial Regression
# 

# %%
from sklearn.preprocessing import PolynomialFeatures
x = df.drop(['charges', 'sex', 'region'], axis = 1)
y = df.charges
pol = PolynomialFeatures (degree = 2)
x_pol = pol.fit_transform(x)
x_train, x_test, y_train, y_test = holdout(x_pol, y, test_size=0.2, random_state=0)
Pol_reg = LinearRegression()
Pol_reg.fit(x_train, y_train)
y_train_pred = Pol_reg.predict(x_train)
y_test_pred = Pol_reg.predict(x_test)
print(Pol_reg.intercept_)
print(Pol_reg.coef_)
print(Pol_reg.score(x_test, y_test))

# %% [markdown]
# Intercept (-5325.88):
# 
# The intercept represents the estimated value of the target variable when all input features are zero. In this context, it's the predicted 'charges' when other factors are not considered.
# Coefficients (Polynomial Regression Terms):
# 
# The coefficients represent the weights assigned to each term in the polynomial features. In your case, these terms include not only the original features but also the combinations of these features up to the second degree (degree=2).
# For example, the coefficient for the term with index 1 corresponds to the linear term for the first feature, and the coefficient for the term with index 12 corresponds to the interaction term between the first and second features.
# R² Score (0.881):
# 
# The R² score is a measure of how well the model explains the variance in the target variable. In this case, the value of 0.881 indicates that around 88.1% of the variance in the 'charges' can be explained by the polynomial regression model.
# In simple terms, the model has learned a polynomial relationship between the input features and the target variable. The intercept and coefficients provide information about the baseline prediction and the importance of each term in the polynomial equation. The high R² score suggests that the model fits the data well, capturing a significant portion of the variability in the charges based on the given features.

# %%
##Evaluating the performance of the algorithm
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)))

# %%
##Predicting the charges
y_test_pred = Pol_reg.predict(x_test)
##Comparing the actual output values with the predicted values
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
df

# %% [markdown]
# Conclusion: like we previously noticed smoking is the greatest factor that affects medical cost charges, then it's bmi and age. Polynomial Regression turned out to be the best model

# %%


# %%



