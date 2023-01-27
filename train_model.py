# Script to train machine learning model.
from ml.data import process_data
import ml.model
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle

if __name__ == '__main__':
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
        test, categorical_features=cat_features, label=None, encoder=encoder, lb=lb, training=False
    )

    # Train and save a model.
    model = ml.model.train_model(X_train, y_train)

    model_paths = {'model': 'model/model.pkl', 'encoder': 'model/encoder.pkl', 'lb': 'model/lb.pkl'}
    for path_key,path in model_paths.items():
        with open(path,'wb') as f:
            pickle.dump(eval(path_key),f)

    # Get metrics performances and save to output text files
    output_paths = {'overall': 'model/output.txt', 'slice': 'model/slice_output.txt'}
    for output_key, output_path in output_paths.items():
        # Slice data performances
        if output_key == 'slice':
            metrics = ml.model.performance_on_slice(model, data=data, cat_features=cat_features, encoder=encoder, lb=lb, label='salary')
        # Overall data performances
        else:
            metrics = ml.model.performance_overall(model, data=test, cat_features=cat_features, encoder=encoder, lb=lb, label='salary')

        # Save to output text
        with open(output_path,'w') as f:
            metrics_str = metrics.to_string(index=False)
            f.write(metrics_str)