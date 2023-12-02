# Flight Delays Prediction Project

![](fd.jpg)
## Overview
The project focuses on predicting flight delays for commercial flights departing from the Washington, D.C., area and arriving in New York in January 2004. The dataset includes information on departure/arrival airports, distances, scheduled times, and the main target variable—whether a flight is delayed (defined as an arrival at least 15 minutes later than scheduled).

## [Phase A: Data Preprocessing ](https://github.com/AdityaDabrase/FlightDelay/blob/main/Flight%20Delay%20EDA.pdf "EDA")

1. **Data Reduction**:
   - Aim: Reduce predictors using domain knowledge or correlation matrices.
   - Result: Save reduced data in "FlightDelaysTrainingData.csv".

2. **Data Exploration**:
   - Procedure: Copy "FlightDelaysTrainingData.csv" to "FlightDelaysDataExploration.csv".
   - Analysis: Generate four Pivot tables to summarize different aspects of the dataset.

3. **Data Conversion**:
   - Conversion: Transform non-numerical data into a suitable format.
   - Reference: Create a table documenting the transformed data.

## [Phase B: Model Building](https://github.com/AdityaDabrase/FlightDelay/blob/main/Flight%20Delay%20Predictions.pdf"Model")
- **Algorithms and Output Types**:
  - Naïve Bayes (NB): Categorical output
  - Classification and Regression Tree (CART): Both categorical and numerical output
  - Logistic Regression: Categorical output

## [Phase C: Using Testing Data](https://github.com/AdityaDabrase/FlightDelay/blob/main/Flight%20Delay%20Predictions.pdf"Model")
- **Generate New Test Data**:
  - Create five new instances and store them in "FlightDelaysTestingData.csv".

## Files in Repository:
- FlightDelaysTrainingData.csv: Reduced dataset after data reduction.
- FlightDelaysDataExploration.csv: Copy of training data for exploration.
- FlightDelaysTestingData.csv: New instances for testing.

## Instructions for Running the Code:
1. Ensure the necessary packages/libraries are installed.
2. Run the respective scripts for each phase of the project.
<!--
## Acknowledgments
- Mention any resources, datasets, or tools used in the project.

## Contributors
- List contributors or authors involved in the project.

## References
- Include any references or sources consulted for the analysis.
