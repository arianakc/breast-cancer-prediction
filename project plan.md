### Descrtion of topics plan to investigate

1. ***10-year survival***. Measure the Receiver-Operating Characteristic Area-Under-Curve (AUC)
   of binary classification of patients surviving longer than 10 years.
2. ***Rank-ordering of events.*** Measure the concordance between predicted risks of patients,
   and the rank-ordering of death events using Concordance Index (CI) 

### Methods and tools plan to use

1. programming language: python3.6, pandas, sklearn

2. Data preprocessing

   1. For clinical data:

      1. Import data by pandas
      2. Deal with categorical data with LabelEncoder from sklearn.preprocessing
      3. Deal with missing data and value with imputing them by following possible values :
         1. A constant value that has meaning within the domain, such as 0, distinct from all other values.
         2. A mean, median or mode value for the column.

   2. For genomic data:

      

3. Baseline methods in sklearn:

   1. Logistic Regression

   2. Support Vector Machines
   3. Decision Tree Algorithm
   4. Random Forest Classification

4. Stretch algorithms:

   1. L1-regularized logistic regression for large datasets 
   2. XGBoost 
   3. Exotic neural networks - something that is not ready-made in TensorFlow 
   4. More ..,

5. Model selection and Validation 

6. Validation



### Timeline for project milestones

1. Project Plan - due 2/18
2. Develop the models and evaluate - 2/18 - 4/10
   1. Finish data preprocessing - 2/18 - 2/28
   2. Finish Baseline methods - 3/1 - 3/10
   3. Finish Strech algorithms - 3/10 - 3/31
   4. Finish Model selection and validation - 4/1 - 4/10
3. Finish the report - 4/10-5/1





