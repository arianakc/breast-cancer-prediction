Group members: Changmao Li, Han He,  Yunze Hao, Caleb Ziems 

### Descrtion of topics plan to investigate

If have time, we will try to generate both classification and regression model.

Classification:

- Class 1 – short-term survivors : 
  - “Died of disease”, time <= 120 mo.
- Class 2 – long-term survivors
  - “Died of disease”, time > 120 mo.
  - “Died of Other Causes”, time > 120 mo.
  -  ”Alive”, time > 120 mo.
- Ignore
  - “Alive”, time <= 120 mo.
  -  “Died of Other Causes”, time <= 120 mo.

Regression:

- produces a score  that increases as survival decreases (risk score)

### Methods and tools plan to use

1. programming language and package: python3.6, pandas, sklearn, Keras, Tensorflow... 

2. Data preprocessing

   1. For clinical data:

      1. Import data by pandas
      2. Deal with categorical data by one-hot encoding
      3. Deal with missing data and value by predicting the missing value if possible otherwise deleting that row

   2. For genomic data:

      1. Import data by pandas
      2. Extract all patient from the tables, make sure the patient is in all three table. Never use fake data for genomic data.
      3. For each patient, extract values from each table, particularly for mutation table, just extract "Consequence" and "Variant_Classification" and concatenate these values together.

3. Baseline methods in sklearn(if have time):

   - DecisionTreeRegressor

   - KNeighborsRegressor
   - ExtraTreesRegressor
   - RandomForestRegressor
   - MLPRegressor
   - TheilSenRegressor
   - RANSACRegressor
   - HuberRegressor
   - LinearRegression
   - ElasticNet
   - GradientBoostingRegressor
   - LarsCV
   - Lasso
   - LassoLarsIC    
   - PassiveAggressiveClassifier
   - LogisticRegression
   - ElasticNetCV
   - OrthogonalMatchingPursuit
   - ARDRegression
   - IsotonicRegression
   - RidgeCV

4. Stretch algorithms:

   1. [L1-regularized logistic regression for large datasets]( https://ai.stanford.edu/~ang/papers/aaai06-l1logisticregression.pdf)  - Caleb Ziems
   2. [XGBoost](https://arxiv.org/abs/1603.02754) - Yunze Hao
   3. [Multilayer perception with grasshopper optimization ](https://www.researchgate.net/profile/Ali_Asghar_Heidari/publication/326692420_An_Efficient_Hybrid_Multilayer_Perceptron_Neural_Network_with_Grasshopper_Optimization/links/5b752129a6fdcc87df804398/An-Efficient-Hybrid-Multilayer-Perceptron-Neural-Network-with-Grasshopper-Optimization.pdf)  -  Changmao Li
   4. [Semi-Supervised Learning](http://pages.cs.wisc.edu/~jerryzhu/pub/ssl_survey.pdf) - Han He

5. Model selection and Validation:
   1. Split train, dev, test as 8:1:1
   2. For classification, measure the Receiver-Operating Characteristic Area-Under-Curve (AUC)
      of binary classification of patients surviving longer than 10 years.
   3. For regression, measure the concordance between predicted risks of patients,
      and the rank-ordering of death events using Concordance Index (CI) 

   

### Timeline for project milestones

1. Project Plan - due 2/18
2. Develop the models and evaluate - 2/18 - 4/10
   1. Finish data preprocessing - 2/18 - 2/28
   2. Finish Baseline methods - 3/1 - 3/15
   3. Finish Strech algorithms - 3/15 - 4/10
3. Finish the report - 4/10-5/1





