# %% [markdown]
# # Logistic Regression Project
# 
# In this project we will be working with advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
# 
# This data set contains the following features:
# 
# * 'Daily Time Spent on Site': consumer time on site in minutes
# * 'Age': cutomer age in years
# * 'Area Income': Avg. Income of geographical area of consumer
# * 'Daily Internet Usage': Avg. minutes a day consumer is on the internet
# * 'Ad Topic Line': Headline of the advertisement
# * 'City': City of consumer
# * 'Male': Whether or not consumer was male
# * 'Country': Country of consumer
# * 'Timestamp': Time at which consumer clicked on Ad or closed window
# * 'Clicked on Ad': 0 or 1 indicated clicking on Ad

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# %% [markdown]
# ## Get the Data
# **Read in the advertising.csv file and set it to a data frame called ad_data.**

# %%
ad_data = pd.read_csv('advertising.csv')

# %% [markdown]
# **Check the head of ad_data**

# %%
ad_data.head()

# %% [markdown]
# ** Use info and describe() on ad_data**

# %%
ad_data.info()

# %%
ad_data.describe()

# %% [markdown]
# ## Exploratory Data Analysis
# 
# Let's use seaborn to explore the data!
# 
# Try recreating the plots shown below!
# 
# ** Create a histogram of the Age**

# %%
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

# %% [markdown]
# **Create a jointplot showing Area Income versus Age.**

# %%
sns.jointplot(x='Age',y='Area Income',data=ad_data)

# %% [markdown]
# **Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age.**

# %%
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde');

# %% [markdown]
# ** Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'**

# %%
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')

# %% [markdown]
# ** Finally, create a pairplot with the hue defined by the 'Clicked on Ad' column feature.**

# %%
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')

# %% [markdown]
# # Logistic Regression
# 
# Now it's time to do a train test split, and train our model!

# %%
from sklearn.model_selection import train_test_split

# %%
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# %% [markdown]
# ** Train and fit a logistic regression model on the training set.**

# %%
from sklearn.linear_model import LogisticRegression

# %%
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# %% [markdown]
# ## Predictions and Evaluations
# 

# %%
predictions = logmodel.predict(X_test)

# %% [markdown]
# ** Create a classification report for the model.**

# %%
from sklearn.metrics import classification_report

# %%
print(classification_report(y_test,predictions))


