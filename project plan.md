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
      2. Deal with categorical data by one-hot encoding
      3. Deal with missing data and value by deleting the samples  with missing value

   2. For genomic data:

      1. Import data by pandas
      2. Extract all patient from the tables, make sure the patient is in all three table. Never use fake data.
      3. For each patient, extract values from each table, particularly for mutation table, just extract "Consequence" and "Variant_Classification" and concatenate these values together.

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

5. Model selection and Validation:

   1. Split train, dev, test as 8:1:1
   2. ...

    

### Timeline for project milestones

1. Project Plan - due 2/18
2. Develop the models and evaluate - 2/18 - 4/10
   1. Finish data preprocessing - 2/18 - 2/28
   2. Finish Baseline methods - 3/1 - 3/15
   3. Finish Strech algorithms - 3/15 - 4/10
3. Finish the report - 4/10-5/1





