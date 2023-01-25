# Udacity Census API


## Overview

This project is to showcase **REST API** with the pretrained `RandomForest Classifier` ML model on the *US Census Income Dataset* as well as **Continuous Integration / Continuous Deployment (CI/CD)** `DevOps` principles. 
For CI step, with **GitHub Actions**, upon deployment, the code itself checks for sanity checks with `PyTest` module. For CD step, the API is being deployed on a cloud platform (**Render**).
The API serves as a forefront to output predictions based on user input (`JSON` data). 


## Model

A RandomForestClassifer with `scikit-learn`'s default hyperparameters other than **random_state(42)** are used. Trained with sklearn version 0.24.1.

### Metrics

To evaluate the model performance, three metrics were used, namely, precision, recall and F1 score. These performance metrics on the model were as follows.

* Precision: 0.74
* Recall: 0.61
* F1 score: 0.67


## Dataset

For dataset, the 1994 US Census's *US Census Income Data Set* is used. For more information on the data, please visit [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income). 

### Training Data

With `scikit-learn`'s `train_test_split` function, 80% of the dataset was used for training purposes. More specifically, 26,048 records were used.


### Evaluation Data

With `scikit-learn`'s `train_test_split` function, 20% of the dataset was used for evaluation purposes. More specifically, 6,513 records were used.


## Continuous Integration (CI)

For continous integration, **GitHub Action** is used. Specifically, upon deployment on the repository, preconfigured **GitHUb Action** runs a series of steps.

### Setting up the environment 

First, the environment under which the code would run is being installed. For this repository, `Python` *3.8* version along with dependencies defined in the `requirements.txt` file are set up.

### Test code

Next, test codes in the **tests** directory are tested with `PyTest` module. Here, only codes starting with test_ are recognized by `PyTest` and are tested automatically. The following files are tested.

* `test_train_model.py`: Testing data preparation and ML model
* `test_api.py`: Testing FastAPI with predefined data inputs

Also, note that `conftest.py` is included for **fixtures** definitions.


## Continuous Deployment (CD)

For continuous deployment on the cloud, [Render](https://render.com) platform is used. Upon a new push on the repository, preconfigured CD runs to set up **REST API**. When no problems are found, the deployment is done and is running.



## REST API

### Implmentation

For **REST API** implementation, `FastAPI` module is used. 
To enable anywhere connectivity, with [Render](https://render.com), the **REST API** is set up on [https://udacity-census-api.onrender.com](https://udacity-census-api.onrender.com).
To allow the data input, **POST** request as well as default **GET** interface are implemented.

* GET: Displays a welcome message
* POST: Returns a prediction of the input data

### Input Data Specifications

The following are specification of each data column.

| Column | Type |
| :---: | :---: |
| age | int |
| workclass | str |
| fnlgt | int |
| education | str |
| education_num | int |
| marital_status | str |
| occupation | str |
| relationship | str |
| race | str |
| sex | str |
| capital_gain | int |
| capital_loss | int |
| hours_per_week | int |
| native_country | str |
| *salary* | Optional[str] |


## Sample API usages

### Default GET method

By accessing the root directory of the API with `GET` method, you are presented with a welcome message.

[!](https://github.com/staedi/udacity-census-api/raw/main/screenshots/live_get.png)

### HTTP 200/OK POST method

To send a `POST` request, unlike its counterpart (`GET`), it should be done with CURL or `requests` module.

[!](https://github.com/staedi/udacity-census-api/raw/main/screenshots/live_post.png)