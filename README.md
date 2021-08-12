# MLBias
Machine Learning Bias project for Data exploration

Project presentation
https://docs.google.com/presentation/d/1GeEUbS7JAYbkb43BDaxqC_CCgXb6GvhBkEHHdzAyv0w/edit?usp=sharing

Project Plan
John Strenio, Jordan Malubay, Yufei Tao

Objective

For this term project, we are going to use supervised machine learning models to find potential racial bias correlations among criminal record datasets with a main goal of identifying and quantifying bias in the dataset. 

Approach

We first had to prep the data, despite it being cleaned by propublica, there's a number of manipulations necessary to take a dataset and run it through a NN, namely shaping the input and encoding non-numerical data for ingestion into a model. We then created a number of visualizations to understand and present the demographics within the data. Finally we built and tested a linear and nonlinear regression model to predict recidivism scores based off of the other attributes. Finally we tested copies of the entire validation set controlling for each race and compared average recidivism scores for each race to identify bias.
	
Team Structure
Because of the size of each individual task, we shared in the implementation as evenly as possible however we each assigned ourselves an area to research and take point on.
	
Preprocessing data: Yufei Tao
ML model related: John Strenio
Result Visualization: Jordan Malubay
Analysis: Everyone in our team

Milestones
Reimplement Propublica’s data filtering for a balanced dataset
Implement 1 or potentially more ML models around the data
Provide clear evidence of known bias (main objective)
Create visualizations that simply and effectively illustrate dataset
Find evidence of potential causes for bias and/or attempt to counter bias
COMPAS DATASET: https://github.com/propublica/compas-analysis
