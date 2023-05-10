# Introdcution
Haberman’s data set contains data from the study conducted in University of Chicago’s Billings Hospital between year 1958 to 1970 for the patients who undergone surgery of breast cancer. Source :https://www.kaggle.com/gilsousa/habermans-survival-data-set)

I would like to explain the various data analysis operation, I have done on this data set and how to conclude or predict survival status of patients who undergone from surgery.

First of all for any data analysis task or for performing operation on data we should have good domain knowledge so that we can relate the data features and also can give accurate conclusion. So, I would like to explain the features of data set and how it affects other feature.

There are 4 attribute in this data set out of which 3 are features and 1 class attribute as below. Also, there are 306 instances of data.

Number of Axillary nodes(Lymph Nodes)
Age
Operation Year
Survival Status
Lymph Node: Lymph nodes are small, bean-shaped organs that act as filters along the lymph fluid channels. As lymph fluid leaves the breast and eventually goes back into the bloodstream, the lymph nodes try to catch and trap cancer cells before they reach other parts of the body. Having cancer cells in the lymph nodes under your arm suggests an increased risk of the cancer spreading.In our data it is axillary nodes detected(0–52)
![image](https://github.com/Anup033/EDA-on-Haberman-dataset/assets/106690260/b41d8ae8-4b35-46ef-ab98-4015d0c50b6b)

Age: It represent the age of patient at which they undergone surgery (age from 30 to 83)

Operation year: Year in which patient was undergone surgery(1958–1969)

Survival Status: It represent whether patient survive more than 5 years or less after undergone through surgery.Here if patients survived 5 years or more is represented as 1 and patients who survived less than 5 years is represented as 2.

References:
https://www.mathworks.com/help/matlab/ref/surfc.html
https://www.breastcancer.org/symptoms/diagnosis/lymph_nodes
https://www.kaggle.com/gilsousa/habermans-survival-data-set
https://www.appliedaicourse.com
