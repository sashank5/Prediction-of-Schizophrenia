import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import randint
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import make_classification
from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from mlxtend.classifier import StackingClassifier
import tensorflow as tf
import argparse

survey_data=pd.read_csv('survey.csv')
# removing unwanted features
survey_data.drop(['comments','state','Country'],axis=1,inplace=True)
missing_data1=survey_data.isnull().sum()
#Now that we have removed unwanted features, lets check for NaN values and replace them
integer = 0
String = 'NaN'
# Create lists by data type
integerFeatures = ['Age']
stringFeatures = ['Gender', 'self_employed', 'work_interfere']
#Find if there are any outliers
for feature in survey_data:
    if feature in integerFeatures:
        survey_data[feature] = survey_data[feature].fillna(integer)
    elif feature in stringFeatures:
        survey_data[feature] = survey_data[feature].fillna(String)
for i in survey_data['Age']:
    if i>100 or i<0:
        survey_data['Age']=survey_data['Age'].replace(i,0)
# Clean the NaN's
mean_age=np.mean(survey_data['Age'])
survey_data['Age']=survey_data['Age'].replace(0,mean_age).round()
gender=survey_data['Gender'].unique()
# As gender is in all different kinds we make sure that only 3 gender available male,female,trans
male = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]
female = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]
for (row, col) in survey_data.iterrows():

    if str.lower(col.Gender) in male:
        survey_data['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female:
        survey_data['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans:
        survey_data['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

#Get rid of unknown
unknown = ['A little about you', 'p']
survey_data = survey_data[~survey_data['Gender'].isin(unknown)]

survey_data['self_employed'] = survey_data['self_employed'].replace(String, 'No')
survey_data['work_interfere'] = survey_data['work_interfere'].replace(String, 'Don\'t know')
# As we have replaced all NaN and null values we check if there are any other missing values
missing_data2=survey_data.isnull().sum()

# We should convert the data into numbers to perform the analysis
# Encoding the data
labels = {}
for feature in survey_data:
    encoder = preprocessing.LabelEncoder()
    encoder.fit(survey_data[feature])
    encoder_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    survey_data[feature] = encoder.transform(survey_data[feature])
    labelKey = 'label_' + feature
    labelValue = [*encoder_mapping]
    labels[labelKey] = labelValue
# to print labels and thier encoded values
#for key, value in labels.items():
 #   print(key, value)
#Covariance testing- Variability comparison between categories of variables
corr_matrix=survey_data.corr().round(2)
f, ax = plt.subplots(figsize=(14, 10))
sns.heatmap(corr_matrix,cmap="YlGnBu", annot=True);

# Distribution and No.of patients by Age
sns.displot(survey_data["Age"], bins=12)
plt.title("Distribution and density by Age")
plt.xlabel("Age")
plt.ylabel("No.of Patients")

# Distribution with no.of patients and treatment
g = sns.FacetGrid(survey_data, col='treatment', height=7)
g.map(sns.distplot, "Age")

#How many people has been treated?
plt.figure(figsize=(12,8))
g = sns.countplot(x="treatment", data=survey_data)
plt.title('Distribution whether treated or not')
# Scaling Age by using MinMaxScalar
scaler = MinMaxScaler()
survey_data['Age'] = scaler.fit_transform(survey_data[['Age']])
# Assigning the required data features for the model to train
features_needed = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
features = survey_data[features_needed]
target = survey_data['treatment']

# split X and y into training and testing sets
batch_size = 100
train_steps = 10000

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(100).repeat().batch(batch_size)

def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

# Define Tensorflow feature columns
age = tf.feature_column.numeric_column("Age")
gender = tf.feature_column.numeric_column("Gender")
family_history = tf.feature_column.numeric_column("family_history")
benefits = tf.feature_column.numeric_column("benefits")
care_options = tf.feature_column.numeric_column("care_options")
anonymity = tf.feature_column.numeric_column("anonymity")
leave = tf.feature_column.numeric_column("leave")
work_interfere = tf.feature_column.numeric_column("work_interfere")
feature_columns = [age, gender, family_history, benefits, care_options, anonymity, leave, work_interfere]

# Build a DNN with 2 hidden layers and 10 nodes in each hidden layer.
model = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[7,10],optimizer=tf.optimizers.Adam,activation_fn=tf.nn.relu)

model.train(input_fn=lambda:train_input_fn(X_train, y_train, batch_size), steps=train_steps)

# Evaluate the model.
eval_result = model.evaluate(input_fn=lambda:eval_input_fn(X_test, y_test, batch_size))

print('\nTest set accuracy: {accuracy:0.2f}\n'.format(**eval_result))
methodDict={}

#Data for final graph
accuracy = eval_result['accuracy'] * 100
methodDict['Neural Network'] = accuracy
predictions = list(model.predict(input_fn=lambda:eval_input_fn(X_train, y_train, batch_size=batch_size)))

# Generate predictions from the model
template = ('\nIndex: "{}", Prediction is "{}" ({:.1f}%), expected "{}"')

# Dictionary for predictions
col1 = []
col2 = []
col3 = []

for idx, input, p in zip(X_train.index, y_train, predictions):
    v  = p["class_ids"][0]
    # Adding to dataframe
    col1.append(idx)  # Index
    col2.append(input)
    col3.append(v)  # Prediction
results = pd.DataFrame({'index': col1, 'expected':col2,'prediction': col3})
print(results.head(10))