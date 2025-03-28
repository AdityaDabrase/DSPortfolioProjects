# %% [markdown]
# ### Analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
# 

# %% [markdown]
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 

# %%
import numpy as np
import pandas as pd

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# %%
df = pd.read_csv('911.csv')

# %%
df.info()

# %%
df.head(5)

# %% [markdown]
# ## BASIC QUESTIONS 

# %% [markdown]
# Top 5 zip codes / townships for 911 calls

# %%
df['zip'].value_counts().head(5)

# %%
df['twp'].value_counts().head(5)

# %%
df['title'].nunique() #how many unique codes?

# %%
#creating a new column for 911 call reason

df['Reason'] = df['title'].apply(lambda title: title.split(':')[0])

# %%
df.head()

# %%
df['Reason'].value_counts()

# %%
sns.countplot(x='Reason',data=df,palette='viridis')

# %% [markdown]
# ## working with timestamp column

# %%
type(df['timeStamp'].iloc[0])

# %%
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

# %%
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

#feature extraction

# %%
dmap = {1:'Mon',2:'Tue',2:'Wed',4:'Thu',5:'Fri',6:'Sat',7:'Sun'}



# %%
df.head()

# %%
sns.countplot(x='Day of Week',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)



# %%
sns.countplot(x='Month',data=df,hue='Reason',palette='viridis')

# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# %%
byMonth = df.groupby('Month').count()
byMonth.head(15)

# %%
# Could be any column
byMonth['twp'].plot()

# %%
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())

# %%
df['Date']=df['timeStamp'].apply(lambda t: t.date())

# %%
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()

# %%
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()

# %%
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()


# %%
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()

# %%
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()

# %%
plt.figure(figsize=(15,10))
sns.heatmap(dayHour,cmap='viridis')

# %%
sns.clustermap(dayHour,cmap='viridis')

# %%
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()

# %%
plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='viridis')

# %%
sns.clustermap(dayMonth,cmap='viridis')


