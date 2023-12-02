# Analyzing 911 Call Data
![alt text](DS-ML/911 calls analysis/dispatch-emergency.gif)


This project involves exploring and analyzing emergency 911 call data, which includes various fields such as latitude, longitude, description, zip code, township, and more. The data covers a wide range of emergency situations and their locations, aiming to extract insights and patterns from the dataset.

## Project Overview

### Data Summary
- The dataset contains 663,522 entries and includes fields like latitude, longitude, description, zipcode, township, and more.
- Data types range from numerical (float, int) to categorical (object) types.

### Basic Analysis
- Identified the top 5 zip codes and townships with the highest number of 911 calls.
- Classified calls based on their reasons (EMS, Traffic, Fire) and determined their counts.

## Exploratory Data Analysis (EDA)

### Time Analysis
- Extracted information from the timestamp column to analyze trends by hour, month, and day of the week.
- Visualized the number of calls based on different time intervals.

### Visualization
- Created various visualizations like count plots, heatmaps, and cluster maps to depict call frequencies and patterns across different time periods and reasons.

## Insights and Findings

- Identified peak times and days for emergency calls.
- Discovered geographical areas with the highest call frequencies.
- Explored trends based on different emergency reasons (EMS, Traffic, Fire).

## Tools Used
- Python Libraries: NumPy, Pandas, Matplotlib, Seaborn
- Data preprocessing, feature extraction, and visualization techniques were utilized.

## Next Steps
- Feature engineering for predictive modelling.
- Applying machine learning algorithms for emergency call prediction.
- Enhancing visualizations for more in-depth analysis.

## Acknowledgments
- The dataset used in this analysis was obtained from Kaggle.

## How to Use
- Clone the repository.
- Run the Jupyter Notebook to explore the code and analysis.
