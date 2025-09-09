# Model Card

# Model Details

# 

# This model is a Logistic Regression classifier implemented using Scikit-learn. It was trained to predict whether an individual's annual income is greater than $50K or not, based on the Census Income dataset.

# 

# Intended Use

# 

# The model is intended for educational purposes as part of the Udacity/WGU Machine Learning DevOps project. It demonstrates how to train, evaluate, and deploy a machine learning model with a FastAPI service and CI/CD pipeline. It is not intended for real-world decision making.

# 

# Training Data

# 

# The training data comes from the UCI Census Income dataset. The dataset includes demographic and employment-related features such as age, education, occupation, and hours worked per week. The dataset was preprocessed with categorical encoding and split into training and testing sets.

# 

# Evaluation Data

# 

# The evaluation data is a held-out test split from the same Census dataset. It was not used during training and provides an unbiased estimate of model performance.

# 

# Metrics

# 

# The model was evaluated using precision, recall, and F1 score.

# 

# Precision: ~0.74

# 

# Recall: ~0.69

# 

# F1 Score: ~0.71

# 

# These results indicate balanced performance between precision and recall.

# 

# Ethical Considerations

# 

# This dataset contains sensitive demographic attributes such as sex, race, and marital status. Using such data in real-world decision-making could reinforce or amplify social biases. Care must be taken to avoid deploying this model in high-stakes environments where fairness and ethics are critical.

# 

# Caveats and Recommendations

# 

# This model is limited to the Census dataset and may not generalize well to other populations or contexts. Its predictions should not be used for making financial, hiring, or policy decisions. Future improvements could include using more robust models, bias mitigation techniques, and updated datasets.

