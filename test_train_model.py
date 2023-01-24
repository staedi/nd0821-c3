import pytest
import numpy as np
from train_model import *

## Fixtures
# Load the data
@pytest.fixture
def data():
    df = pd.read_csv('data/census.csv')
    return df

# Split train/test data
@pytest.fixture
def data_split(data):
    def _data_split(test_size=0.2):
        train, test = train_test_split(data, test_size=test_size)
        return data, train, test
    
    return _data_split

# Preprocess data
@pytest.fixture
def prepare_data(data_split):
    data, train, test = data_split()
    def _prepare_data(data_type,encoder=None,lb=None,label='salary'):
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
@pytest.fixture
def prepare_model(prepare_data):
    X_train, y_train, encoder, lb = prepare_data('train')
    X_test, y_test, encoder, lb = prepare_data('test', encoder=encoder, lb=lb)
    model = ml.model.train_model(X_train,y_train)

    def _prepare_model(data_type):
        if data_type == 'train':
            return model, X_train, y_train
        else:
            return model, X_test, y_test

    return _prepare_model

## Tests
# Data validity
def test_data(data):
    columns = data.columns.tolist()
    assert len(data) > 0
    assert len(columns) == len(list(filter(lambda x:x.strip()==x,columns)))
    assert all(data == data.applymap(lambda x:x.strip() if isinstance(x,str) else x))

# Data split
def test_train_test_split(data_split):
    data, train, test = data_split(0.2)
    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(data)
    assert len(test) == np.ceil(len(data)/5)

# Preprocess data
def test_process_data(prepare_data):
    X_train, y_train, encoder, lb = prepare_data('train')
    assert len(X_train)>0
    assert len(y_train)>0
    assert encoder is not None
    assert lb is not None
    X_test, y_test, encoder, lb = prepare_data('test', encoder=encoder, lb=lb)
    assert len(X_test)>0
    assert len(y_test)>0
    assert encoder is not None
    assert lb is not None

# Train model
def test_train_model(prepare_model):
    model, X_test, y_test = prepare_model('test')
    assert model is not None
    assert len(list(filter(lambda x:isinstance(x,np.int64),model.predict(X_test)))) == len(X_test)

# Model metrics
def test_compute_metrics(prepare_model):
    model, X_test, y_test = prepare_model('test')
    preds = model.predict(X_test)
    precision, recall, fbeta = ml.model.compute_model_metrics(y_test, preds)
    assert isinstance(precision,float)
    assert isinstance(recall,float)
    assert isinstance(fbeta,float)
    assert precision >= 0.7
    assert recall >= 0.5
    assert fbeta >= 0.6

# Model inference
def test_inference(prepare_model):
    model, X_train, y_train = prepare_model('train')
    assert all(model.predict(X_train)) == all(ml.model.inference(model,X_train))