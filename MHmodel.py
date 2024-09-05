import pickle
import re
from dateutil import parser
import numpy as np
import pandas as pd
import altair as alt
import missingno as msn
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BicScore, HillClimbSearch, BayesianEstimator
from pgmpy.inference import VariableElimination
from imblearn.over_sampling import SMOTE
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from dateutil import parser
import plotly.graph_objects as go
import plotly.subplots as sp
import graphviz
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from graphviz import Digraph
from IPython.display import Image
import matplotlib.image as mpimg
def fill_missing_values_with_group_mean_or_mode(dataset, targetColumn: str) -> pd.DataFrame:
    # Check if the target column is numeric
    if pd.api.types.is_numeric_dtype(dataset[targetColumn]):
        # Fill missing values with the mean of the group
        dataset[targetColumn] = dataset.groupby('Fav genre')[targetColumn].transform(
            lambda x: x.fillna(x.mean().round()))
    else:
        # Fill missing values with the mode of the group
        dataset[targetColumn] = dataset.groupby('Fav genre')[targetColumn].transform(
            lambda x: x.fillna(x.mode().iloc[0]))

    return dataset

def fill_missing_values_by_columns(dataset, columnNames):
    # Iterate over the list of column names to process each column
    for columnName in columnNames:
        # Check if that column has any empty value
        if dataset[columnName].isna().any():
            # Fill missing values by applying the function to calculate and fill mean or mode by favorite music genre
            dataset = fill_missing_values_with_group_mean_or_mode(dataset, columnName)

    return dataset

dataset = pd.read_csv("C:/Users/15154/Desktop/mxmh_survey_results.csv")
# EDA before data preprocessing tells us that BPM is indeed a musical feature with the most missing values and needs to be optimized. And we can't simply delete lines with missing values, because those lines may contain some very useful information that can be exploited.
# First, we first identify the numerical data in all columns, because only they may have outliers from the column average. Like the previous EDA, we use a boxplot that can reflect the data quartile. It turns out that one column not only has a much higher value than the rest of the data in that column, but that value also breaks the measure of the feature,
# the column is BPM, which makes sense since it was previously known to have the most missing values

print(dataset["BPM"].max())
print(dataset["BPM"].min())
print(dataset.loc[dataset["BPM"] == dataset["BPM"].max()]["BPM"])
print(dataset.loc[dataset["BPM"] == dataset["BPM"].min()]["BPM"])
#Through the previous EDA analysis, we decided to keep the BPM data in the range of 20-500, and for the data outside the range, I did not plan to delete them, but replaced these values with the average of other rows
def replace_outliers_with_mean_bpm(dataset):
    valid_bpm_mean = dataset[(dataset['BPM'] >= 20) & (dataset['BPM'] <= 500)]['BPM'].mean()
    outlier_indices = dataset[(dataset['BPM'] < 20) | (dataset['BPM'] > 500)].index
    dataset.loc[outlier_indices, 'BPM'] = valid_bpm_mean

    return dataset
dataset = replace_outliers_with_mean_bpm(dataset)
#def plot_boxplot_of_bpm(dataset):
    #plt.figure(figsize=(10, 6))
    #sns.boxplot(x=dataset['BPM'])
    #plt.title('Boxplot of BPM')
    #plt.xlabel('BPM')
    #plt.show()

#plot_boxplot_of_bpm(dataset)
#Looking again at the BPM distribution is much more normal
columnNames = dataset.columns.tolist()
newDataSet = fill_missing_values_by_columns(dataset=dataset, columnNames=columnNames)
#format data
def validate_and_convert_types(dataset, column_types):
    for column, dtype in column_types.items():
        try:
            dataset[column] = dataset[column].astype(dtype)
            print(f"{column} converted successfully to {dtype}.")
        except ValueError as e:
            print(f"Error converting {column} to {dtype}: {e}")

    for column in dataset.select_dtypes(include=['number']).columns:
        if pd.to_numeric(dataset[column], errors="coerce").notna().all():
            print(f"{column} is fully numeric.")
        else:
            print(f"{column} contains non-numeric values.")

    if "Timestamp" in dataset.columns:
        dataset["Timestamp"] = dataset["Timestamp"].apply(lambda x: parser.parse(x, dayfirst=True))
        print("Timestamp converted to datetime.")

    return dataset

column_types = {
    "Age": 'int32',
    "Primary streaming service": 'str',
    "Hours per day": 'float',
    "While working": 'str',
    "Instrumentalist": 'str',
    "Composer": 'str',
    "Fav genre": 'str',
    "Exploratory": 'str',
    "Foreign languages": 'str',
    "BPM": 'int32',
    "Frequency [Classical]": 'str',
    "Frequency [Country]": str,
    "Frequency [EDM]": str,
    "Frequency [Folk]": str,
    "Frequency [Gospel]": str,
    "Frequency [Hip hop]": str,
    "Frequency [Jazz]": str,
    "Frequency [K pop]": str,
    "Frequency [Latin]": str,
    "Frequency [Lofi]": str,
    "Frequency [Metal]": str,
    "Frequency [Pop]": str,
    "Frequency [R&B]": str,
    "Frequency [Rap]": str,
    "Frequency [Rock]": str,
    "Frequency [Video game music]": str,
    "Anxiety": 'int32',
    "Depression": 'int32',
    "Insomnia": 'int32',
    "OCD": 'int32',
    "Music effects": 'str',
    "Permissions": 'str'
}

dataset = validate_and_convert_types(newDataSet, column_types)
dataset.info()
#We need to clear out some variables that are clearly not related to mental health
unnecessaryColumns = ["Primary streaming service", "Exploratory", "Foreign languages", "Permissions"]
dataset.drop(unnecessaryColumns, axis=1, inplace=True)

# Next is the model building process. We select age, hours of listening to music per day, BPM, major streaming services, anxiety score, depression score, favorite type of music as the characteristic variables, and "whether health status has improved" as the target variable.
# From the previous EDA analysis, it can be seen that age and mental health have a certain correlation, which is mainly reflected in anxiety and depression, two mental health problems.
# Time spent listening to music per day also had a stronger relationship with mental health, especially for people with mild and severe mental health problems,
#Although BPM is not closely related to the score of anxiety and other psychological problems, as one of the most important characteristics of music, we still guess that it has a certain relationship with the improvement effect of mental health, so we choose it
# Favorite music genre as auxiliary feature variable
# For the score of the four mental health problems, I choose anxiety and depression to add characteristic variables, because these two problems are most closely related to mental health, while insomnia is related to both of these psychological problems. OCD appears at random, and it seems that it is not closely related to music factors.
# Target variable
dataset0=dataset.copy()
dataset1=dataset.copy()
dataset2=dataset.copy()
dataset3=dataset.copy()
dataset4=dataset.copy()
dataset5=dataset.copy()
dataset6=dataset.copy()
dataset['Improved_Health'] = (dataset['Music effects'] == 'Improve').astype(int)
#feature variables
features = ['Age', 'Hours per day', 'BPM',  'Anxiety', 'Depression', 'Fav genre']
X = dataset[features]
y = dataset['Improved_Health']
#Encoding categorical variables, using One-hot Encoding, which is a technique that converts categorical variables into unique thermal encoding, its mechanism is to identify the unique value of the categorical variable, and then create a binary column for each category, and finally fill the column of the corresponding category with 1, and the remaining columns with 0. So that categorical variables can be better applied to linear models.
categorical_columns = ['Fav genre']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(X[categorical_columns])
encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
X = X.drop(columns=categorical_columns)
X = pd.concat([X.reset_index(drop=True), encoded_features_df], axis=1)
#Training Set & Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# SMOTE up-sampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# standardize numerical features
scaler = StandardScaler()
X_train_resampled[['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression']] = scaler.fit_transform(X_train_resampled[['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression']])
X_test[['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression']] = scaler.transform(X_test[['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression']])

# modelling
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# Define grid search parameters
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear']
}

# Hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_resampled, y_train_resampled)

# Output optimal parameters
print("Best parameters found: ", grid_search.best_params_)

# Train the model using optimal parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_resampled, y_train_resampled)

# prediction
y_pred = best_model.predict(X_test)

# model evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# ROC curve and AUC
y_prob = best_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

#Random Forest

dataset1['Improved_Health'] = (dataset1['Music effects'] == 'Improve').astype(int)

features = ['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression', 'Fav genre']
X = dataset1[features]
y = dataset1['Improved_Health']

# Encode categorical variables using OneHotEncoder
categorical_columns = ['Fav genre']
encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_features = encoder.fit_transform(X[categorical_columns])

# Create a new feature data bo
encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))

# Combine numerical features with encoded classification features
X = X.drop(columns=categorical_columns)
X = pd.concat([X.reset_index(drop=True), encoded_features_df], axis=1)

# Split the data into training sets and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardized numerical features
scaler = StandardScaler()
X_train[['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression']] = scaler.fit_transform(X_train[['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression']])
X_test[['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression']] = scaler.transform(X_test[['Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression']])

# Upsample to balance the categories
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Define a random forest mode
rf_model = RandomForestClassifier(random_state=42)

# Define random search parameters
param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Use RandomizedSearchCV for hyperparameter adjustment
random_search_rf = RandomizedSearchCV(estimator=rf_model, param_distributions=param_dist_rf, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search_rf.fit(X_train_resampled, y_train_resampled)

# Output optimal parameters
print("Best parameters found: ", random_search_rf.best_params_)

# Use optimal parameters to train the mode
best_rf_model = random_search_rf.best_estimator_
best_rf_model.fit(X_train_resampled, y_train_resampled)

# prediction
y_pred_rf = best_rf_model.predict(X_test)

# model evaluation
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')
print(f'Random Forest Confusion Matrix:\n{conf_matrix_rf}')
print(f'Random Forest Classification Report:\n{class_report_rf}')

# ROC curve and AUC
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# I used two algorithms, logistic regression and random forest,
# The model accuracy before hyperparameter adjustment and class imbalance is about 75%, but the F1-score of class 0 (no improvement in health) is much lower than that of class 1 (improvement in health)
# This means that the model's prediction of class 0 is not accurate, and the prediction is more biased toward class 1, which is very understandable, because class 1 accounted for about 75% of the original data set
# For this I tried to adjust the model parameter setting weights or do up sampling (SMOTE) to balance the number of inter-class samples, and do grid search or random search two hyperparameter adjustment methods
# But the best result is simply that the model has an AUC value of 0.59, slightly higher than 0.5, which indicates that the model is weak in distinguishing between positive and negative classes
# Given the large gap in the ratio of positive and negative in the data set, "the effect of music on health improvement" is not a good target variable, so we choose other variables
# Based on the previous EDA analysis, the mean anxiety score was 5.84 and the median was 6, so it is appropriate that we delineate the criteria for high anxiety as "anxiety score greater than 6"
# "High_Anxiety" (anxiety score >6) was selected as the target variable, and age, hours of listening to music per day, BPM, depression score, insomnia score and favorite type of music were selected as the characteristic variables to establish a logistic regression model
dataset2['High_Anxiety'] = (dataset2['Anxiety'] > 6).astype(int)
features_for_select = ['Age', 'Hours per day', 'BPM', 'Fav genre','Instrumentalist','Composer','While working','Depression','Insomnia']
X = dataset2[features_for_select]
y = dataset2['High_Anxiety']
categorical_columns = ['Fav genre', 'Instrumentalist', 'Composer','While working']
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
scaler = StandardScaler()
X[['Age', 'Hours per day', 'BPM','Depression','Insomnia']] = scaler.fit_transform(X[['Age', 'Hours per day', 'BPM','Depression','Insomnia']])
model_for_select = LogisticRegression(max_iter=1000, class_weight='balanced')
rfe = RFE(estimator=model_for_select, n_features_to_select=5)
rfe.fit(X,y)

ranking=rfe.ranking_
feature_ranking = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': ranking
})
feature_ranking.sort_values(by='Ranking', inplace=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_ranking['Feature'], feature_ranking['Ranking'], color='skyblue')
plt.xlabel('Ranking')
plt.title('Feature Importance Ranking')
plt.gca().invert_yaxis()
plt.show()

selected_features = X.columns[rfe.support_]
print("Selected features: ", selected_features)
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF
X_selected = X[selected_features]
X_const = sm.add_constant(X_selected)
# VIF of features
vif = pd.DataFrame()
vif['Feature'] = X_selected.columns
vif['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(1, X_const.shape[1])]

print(vif)

#Logistic Regression 2
dataset5['High_Anxiety'] = (dataset5['Anxiety'] > 6).astype(int)

# Contains the feature variables that need to be selected
features = ['Age', 'Hours per day', 'While working', 'Fav genre', 'Instrumentalist', 'Composer','Depression','Insomnia']
X = dataset5[features]
y = dataset5['High_Anxiety']

# Label Encoding
label_encoder = LabelEncoder()
for col in ['Fav genre','Instrumentalist', 'Composer','While working']:
    X[col] = label_encoder.fit_transform(X[col])

# split data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the processing of numerical data
scaler = StandardScaler()
X_train[['Age', 'Hours per day', 'Depression','Insomnia']] = scaler.fit_transform(X_train[['Age', 'Hours per day', 'Depression','Insomnia']])
X_test[['Age', 'Hours per day', 'Depression','Insomnia']] = scaler.transform(X_test[['Age', 'Hours per day', 'Depression','Insomnia']])

# LR2
log_reg_model2 = LogisticRegression(max_iter=1000, class_weight='balanced')

# random search parameters
param_dist_lr = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['newton-cg', 'lbfgs', 'liblinear']
}

random_search_lr = RandomizedSearchCV(estimator=log_reg_model2, param_distributions=param_dist_lr, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search_lr.fit(X_train, y_train)

print("Best parameters found: ", random_search_lr.best_params_)
best_lr_model = random_search_lr.best_estimator_
best_lr_model.fit(X_train, y_train)

# prediction
y_pred_lr = best_lr_model.predict(X_test)

# model evaluation
accuracy_lr = accuracy_score(y_test, y_pred_lr)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
class_report_lr = classification_report(y_test, y_pred_lr)
print(f'Logistic Regression Accuracy: {accuracy_lr}')
print(f'Logistic Regression Confusion Matrix:\n{conf_matrix_lr}')
print(f'Logistic Regression Classification Report:\n{class_report_lr}')

# get coefficients
coefficients = best_lr_model.coef_[0]
intercept = best_lr_model.intercept_

# Displays intercept and coefficien
print(f"Intercept: {intercept}")
print("Coefficients:")
for feature, coef in zip(X.columns, coefficients):
    print(f"{feature}: {coef}")

# odd ratios
odds_ratios = np.exp(coefficients)

print("Odds Ratios:")
for feature, odds_ratio in zip(X.columns, odds_ratios):
    print(f"{feature}: {odds_ratio}")

# Add a constant term (intercept) to the model
X_train_sm = sm.add_constant(X_train)

# use statsmodels for regression again
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()

print(result.summary())

# ROC curve and AUC
y_prob_lr = best_lr_model.predict_proba(X_test)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

plt.figure()
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_lr)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
# After selecting high anxiety level as the target variable, the accuracy of the model came to 72.97%.
#The accuracy rate of class #0 is 0.69, the recall rate is 0.75, the recall rate of class 1 is 0.77, the recall rate is 0.71, and the F1-score of both are 0.72 and 0.74, respectively.
# The AUC value of this model is 0.78, indicating that the model is good at distinguishing between the positive class (high anxiety) and the negative class (not high anxiety)
# Random Forest 2
rf_model2 = RandomForestClassifier(random_state=42, class_weight='balanced')

param_dist_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

random_search_rf2 = RandomizedSearchCV(estimator=rf_model2, param_distributions=param_dist_rf, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search_rf2.fit(X_train, y_train)

print("Best parameters found: ", random_search_rf2.best_params_)
best_rf_model = random_search_rf2.best_estimator_
best_rf_model.fit(X_train, y_train)

y_pred_rf = best_rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)
print(f'Random Forest Accuracy: {accuracy_rf}')
print(f'Random Forest Confusion Matrix:\n{conf_matrix_rf}')
print(f'Random Forest Classification Report:\n{class_report_rf}')

# ROC curve and AUC
y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# All indicators of the random forest model obtained here (accuracy, class F1-score and AUC) are inferior to the previous logistic regression model, which indicates that logistic regression performs slightly better on this data set, possibly because the feature relationship is more linear, so despite the higher complexity and nonlinear capturing ability of the random forest model, The performance is still not significantly beyond logistic regression.
# Although I got two well-performing classification models, it seems that using one mental state to evaluate the other mental state lacks some practical value. I would like to add more music-related variables to predict a particular degree of mental health problems, and hope to better understand the causality and other connections between the variables
# The ultimate goal is to build a model that can be used in practice. People can know whether they are in a relatively high depression state by inputting some of their states (including information such as music frequency). Bayesian network model is very suitable for this task, because the dataset we selected is smaller in size, Bayes performs better, and it can better explain the relationship between variables. Leverage smaller computing resources
# Bayesian mode
dataset4['High_Anxiety'] = (dataset4['Anxiety'] > 6).astype(int)
dataset4['High_Depression'] = (dataset4['Depression'] > 5).astype(int)
dataset4['Hours per day'] = dataset4['Hours per day'].round().astype(int)
label_encoder = LabelEncoder()
dataset4['Fav genre'] = label_encoder.fit_transform(dataset4['Fav genre'])
dataset4['Music effects'] = label_encoder.fit_transform(dataset4['Music effects'])
dataset4['Instrumentalist'] = label_encoder.fit_transform(dataset4['Instrumentalist'])
dataset4['Composer'] = label_encoder.fit_transform(dataset4['Composer'])
frequency_columns1 = [
        'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]','Frequency [Folk]',
        'Frequency [Hip hop]', 'Frequency [Jazz]', 'Frequency [K pop]','Frequency [Gospel]',
        'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]',
        'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]'
    ]
for col in frequency_columns1:
    dataset4[col] = label_encoder.fit_transform(dataset4[col])
variables = [
    'Age', 'Hours per day', 'Fav genre', 'BPM','Instrumentalist','Composer',
    'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]', 'Frequency [Folk]',
    'Frequency [Hip hop]', 'Frequency [Jazz]', 'Frequency [K pop]', 'Frequency [Gospel]',
    'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]',
    'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]',
    'Anxiety', 'Depression','OCD','Insomnia','High_Anxiety', 'High_Depression', 'Music effects'
]
data = dataset4[variables]

hc = HillClimbSearch(data)
best_model = hc.estimate(scoring_method=BicScore(data))
best_model.add_edge('Instrumentalist', 'High_Anxiety')
print("Network edges:", best_model.edges())
#plot
dot = graphviz.Digraph()
for node in best_model.nodes():
    dot.node(node, shape="ellipse", style="filled", color="lightblue2")

for edge in best_model.edges():
    dot.edge(edge[0], edge[1])

import os
save_directory = 'G:/Research/plots'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

dot.render(os.path.join(save_directory, 'bayesian_network_circular'), format='png', cleanup=True)

print(f"already saved to {os.path.join(save_directory, 'bayesian_network_circular.png')}")
def preprocess_data(dataset):
    # Add target variable 'High_Anxiety'
    dataset['High_Anxiety'] = (dataset['Anxiety'] > 6).astype(int)
    dataset['High_Depression'] = (dataset['Depression'] > 5).astype(int)
    # Encode 'Fav genre' as categorical variable
    label_encoder = LabelEncoder()
    dataset['Fav genre'] = label_encoder.fit_transform(dataset['Fav genre'])
    dataset['Music effects'] = label_encoder.fit_transform(dataset['Music effects'])
    dataset['Instrumentalist'] = label_encoder.fit_transform(dataset['Instrumentalist'])
    dataset['Composer'] = label_encoder.fit_transform(dataset['Composer'])
    # Encode frequency columns
    frequency_columns = [
        'Frequency [Classical]', 'Frequency [Country]', 'Frequency [EDM]','Frequency [Folk]',
        'Frequency [Hip hop]', 'Frequency [Jazz]', 'Frequency [K pop]',
        'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Metal]', 'Frequency [Pop]',
        'Frequency [R&B]', 'Frequency [Rap]', 'Frequency [Rock]', 'Frequency [Video game music]'
    ]

    for col in frequency_columns:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    return dataset, frequency_columns
# Apply preprocessing
dataset3, frequency_columns = preprocess_data(dataset3)

print(dataset3.info())
print(f"预处理后的数据集形状: {dataset3.shape}")
# Features and target variable
features = ['Fav genre','Anxiety','Instrumentalist','Composer'] + frequency_columns
target = 'High_Depression'
dataset3=dataset3[features + [target]]

train_data, test_data = train_test_split(dataset3, test_size=0.2, random_state=42)
print(train_data.shape)
print(test_data.shape)
##
bic=BicScore(train_data)
hc=HillClimbSearch(train_data)

best_model=hc.estimate(scoring_method=BicScore(train_data))

expected_edges = [
    ('Frequency [Rock]','High_Depression'),
    ('Frequency [Rap]','High_Depression'),
    ('Instrumentalist','Anxiety')
]

for edge in expected_edges:
    if edge[0] in best_model.nodes() and edge[1] in best_model.nodes():
        best_model.add_edge(*edge)
    else:
        print(f"Edge {edge} contains nodes that are not in the graph")

# Print the best model edges
print("Best model edges: ", best_model.edges())
dot = graphviz.Digraph()
for node in best_model.nodes():
    dot.node(node, shape="ellipse", style="filled", color="lightblue2")
for edge in best_model.edges():
    dot.edge(edge[0], edge[1])
dot.render("bayesian_network", format="png", cleanup=True)
dot.view()
model = BayesianNetwork(best_model.edges())
model.fit(train_data, estimator=BayesianEstimator)

# save the model
model_path = 'G:/Research/Models/bayesian_network_model.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_path}")

predictions=model.predict(test_data.drop("High_Depression",axis=1))
# Evaluate the model performance
accuracy = accuracy_score(test_data["High_Depression"], predictions)
conf_matrix = confusion_matrix(test_data["High_Depression"], predictions)
class_report = classification_report(test_data["High_Depression"], predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# After continuous selection and deletion of nodes and addition of custom network structure, I obtained the highest level of integration of Bayesian network model (classification task) in this data set, with an accuracy of 67%, which is in line with expectations
#Using this model, we can input our music genre preferences and a range of music listening frequencies to make the most basic judgments about our depression.
# In addition, after the experiment, I found that the performance of this data set was poor. From the stage of EDA analysis, I found that there was a lot of noise in this data set, and even after data processing, the connection between variables was not satisfactory, and the connection between music data and mental health data could not be well presented in the machine learning model, which may be more suitable for simple linear fitting
# Therefore, we cannot further analyze this dataset to obtain other mental health information to understand the relationship between music and mental health. However, we can obtain a lot of useful information through EDA, which can be used as a useful reference for future experiments and data collection, and, like the data politics we considered before the study, whether music data is related to mental health depends on many factors
#For example, the religious belief and economic level of the respondents, which has great implications for the data collection stage.
import pickle
model_path = 'G:/Research/Models/bayesian_network_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
# The serialization model is a string
model_str = pickle.dumps(model).hex()
print(model_str)
def preprocess_data2(dataset):
    # Add target variable 'High_Anxiety'
    dataset['High_Anxiety'] = (dataset['Anxiety'] > 6).astype(int)
    dataset['High_Depression'] = (dataset['Depression'] > 5).astype(int)
    # Encode 'Fav genre' as categorical variable
    label_encoder = LabelEncoder()
    dataset['Fav genre'] = label_encoder.fit_transform(dataset['Fav genre'])
    dataset['Music effects'] = label_encoder.fit_transform(dataset['Music effects'])
    dataset['Instrumentalist'] = label_encoder.fit_transform(dataset['Instrumentalist'])
    dataset['Composer'] = label_encoder.fit_transform(dataset['Composer'])
    # Encode frequency columns
    frequency_columns2 = [
        'Frequency [Classical]',
        'Frequency [Hip hop]', 'Frequency [Jazz]',
        'Frequency [Latin]', 'Frequency [Lofi]', 'Frequency [Pop]',
        'Frequency [R&B]', 'Frequency [Rap]',  'Frequency [Video game music]'
    ]

    for col in frequency_columns2:
        dataset[col] = label_encoder.fit_transform(dataset[col])

    return dataset, frequency_columns2
# Apply preprocessing
dataset6, frequency_columns2 = preprocess_data2(dataset6)

print(dataset6.info())
print(f"预处理后的数据集形状: {dataset6.shape}")
# Features and target variable
features = ['Anxiety','Instrumentalist','Composer'] + frequency_columns2
target = 'High_Depression'
dataset6=dataset6[features + [target]]

train_data2, test_data2 = train_test_split(dataset6, test_size=0.2, random_state=123)
print(train_data2.shape)
print(test_data2.shape)
##
bic=BicScore(train_data2)
hc=HillClimbSearch(train_data2)

best_model2=hc.estimate(scoring_method=BicScore(train_data2))

expected_edges2 = [
    ('Frequency [Rock]','High_Depression'),
    ('Frequency [Rap]','High_Depression'),
    ('Instrumentalist','Anxiety')
]
for edge in expected_edges2:
    if edge[0] in best_model2.nodes() and edge[1] in best_model2.nodes():
        best_model2.add_edge(*edge)
    else:
        print(f"Edge {edge} contains nodes that are not in the graph")

# Print the best model edges
print("Best model edges: ", best_model2.edges())
dot = graphviz.Digraph()
for node in best_model2.nodes():
    dot.node(node, shape="ellipse", style="filled", color="lightblue2")
for edge in best_model2.edges():
    dot.edge(edge[0], edge[1])
dot.render("bayesian_network", format="png", cleanup=True)
dot.view()
model2 = BayesianNetwork(best_model2.edges())
model2.fit(train_data2, estimator=BayesianEstimator)


predictions2=model2.predict(test_data2.drop("High_Depression",axis=1))
# Evaluate the model performance
accuracy2 = accuracy_score(test_data2["High_Depression"], predictions2)
conf_matrix2 = confusion_matrix(test_data2["High_Depression"], predictions2)
class_report2 = classification_report(test_data2["High_Depression"], predictions2)

print(f'Accuracy: {accuracy2}')
print(f'Confusion Matrix:\n{conf_matrix2}')
print(f'Classification Report:\n{class_report2}')
