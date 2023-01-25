import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model

## Fixtures for model 
# Load the data
@pytest.fixture(scope='session')
def data():
    df = pd.read_csv('data/census.csv')
    return df

# Split train/test data
@pytest.fixture(scope='session')
def data_split(data):
    def _data_split(test_size=0.2):
        train, test = train_test_split(data, test_size=test_size)
        return data, train, test

    return _data_split

# Preprocess data
@pytest.fixture(scope='session')
def prepare_data(data_split):
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

    data, train, test = data_split()

    def _prepare_data(data_type, encoder=None, lb=None, label='salary'):
        if data_type == 'train':
            training = True
            X_train, y_train, encoder, lb = process_data(train, categorical_features=cat_features, label=label, training=training)
            return X_train, y_train, encoder, lb
        else:
            training = False
            X_test, y_test, encoder, lb = process_data(test, categorical_features=cat_features, label=label, encoder=encoder, lb=lb, training=training)
            return X_test, y_test, encoder, lb

    return _prepare_data

# Train model
@pytest.fixture(scope='session')
def prepare_model(prepare_data):
    X_train, y_train, encoder, lb = prepare_data('train')
    X_test, y_test, encoder, lb = prepare_data('test', encoder=encoder, lb=lb)
    model = train_model(X_train, y_train)

    def _prepare_model(data_type):
        if data_type == 'train':
            return model, X_train, y_train
        else:
            return model, X_test, y_test

    return _prepare_model


## Fixtures for API
# Over 50K
@pytest.fixture(scope='session')
def payload_over_sample():
    payload = {
        "age": 31,
        "workclass": "Private",
        "fnlgt": 45781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-speciality",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital-gain": 14000,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": "United-States",
    }

    return payload

# Under 50K
@pytest.fixture(scope='session')
def payload_under_sample():
    payload = {
        "age": 66,
        "workclass": "Private",
        "fnlgt": 211781,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Never-married",
        "occupation": "Prof-speciality",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": "United-States",
    }

    return payload

# Errorneous
@pytest.fixture(scope='session')
def payload_error_sample():
    payload = {
        "age": 66,
        "workclass": "Private",
        "fnlgt": 211781,
        "education": "Masters",
        "education-num": "Twenty",
        "marital-status": "Never-married",
        "occupation": "Prof-speciality",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 55,
        "native-country": "United-States",
    }

    return payload