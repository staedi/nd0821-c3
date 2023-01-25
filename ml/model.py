from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ml.data import process_data
import pandas as pd

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)
    return preds


def performance_overall(model, data, cat_features, encoder, lb, label='salary'):
    """ Evalue performance on test data.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `cat_features`
    cat_features: list[str]
        List containing the names of the categorical features
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)

    Returns
    -------
    metrics_df : np.array
        Metrics of test data from the model.
    """

    metrics = []

    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label=label, encoder=encoder, lb=lb, training=False)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    metrics.append((precision, recall, fbeta))
    
    metrics_df = pd.DataFrame(metrics, columns=['precision', 'recall', 'f1'])
    metrics_mean = metrics_df[['precision', 'recall', 'f1']].mean()
    return metrics_df, metrics_mean


def performance_on_slice(model, data, cat_features, encoder, lb, label='salary'):
    """ Evalue performance on sliced data.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in `cat_features`
    cat_features: list[str]
        List containing the names of the categorical features
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    label : str
        Name of the label column in `X`. If None, then an empty array will be returned
        for y (default=None)

    Returns
    -------
    metrics_df : np.array
        Metrics of slices from the model.
    """

    metrics = []

    for cat in cat_features:
        unique_values = data[cat].unique()
        for value in unique_values:
            X, y, encoder, lb = process_data(data.loc[data[cat]==value], categorical_features=cat_features, label=label, encoder=encoder, lb=lb, training=False)
            preds = inference(model, X)
            precision, recall, fbeta = compute_model_metrics(y, preds)
            metrics.append((cat, value, precision, recall, fbeta))
    
    metrics_df = pd.DataFrame(metrics, columns=['category', 'category_value', 'precision', 'recall', 'f1'])
    metrics_mean = metrics_df[['precision', 'recall', 'f1']].mean()
    return metrics_df, metrics_mean
    