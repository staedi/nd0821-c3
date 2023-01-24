# Script to train machine learning model.
from ml.data import process_data
import ml.model
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle

# Add code to load in the data.
data = pd.read_csv('data/census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", encoder=encoder, lb=lb, training=False
)

# Train and save a model.
model = ml.model.train_model(X_train, y_train)

model_paths = {'model':'model/model.pkl','encoder':'model/encoder.pkl','lb':'model/lb.pkl'}
for path_key,path in model_paths.items():
    with open(path,'wb') as f:
        pickle.dump(eval(path_key),f)

# Slice data performances
metrics, metrics_mean = ml.model.performance_on_slice(model, data=data, cat_features=cat_features, encoder=encoder, lb=lb, label='salary')