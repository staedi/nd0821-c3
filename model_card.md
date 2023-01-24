# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf


## Model Details

A RandomForestClassifer with `scikit-learn`'s default hyperparameters other than **random_state(42)** are used. Trained with sklearn version 0.24.1.
For dataset, the 1994 US Census's *US Census Income Data Set* is used. For more information on the data, please visit [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/census+income). 


## Intended Use

This model is to predict whether income salary is larger than $50,000. 


## Training Data

With `scikit-learn`'s `train_test_split` function, 80% of the dataset was used for training purposes. More specifically, 26,048 records were used.


## Evaluation Data

With `scikit-learn`'s `train_test_split` function, 20% of the dataset was used for evaluation purposes. More specifically, 6,513 records were used.


## Metrics

To evaluate the model performance, three metrics were used, namely, precision, recall and F1 score. These performance metrics on the model were as follows.

* Precision: 0.74
* Recall: 0.61
* F1 score: 0.67


## Ethical Considerations

Along with other census data, this specific data was surveyed and collected from people. As it happens, no one can be sure about the completeness of the census as it might involve biases by surveyors or from the people who are being surveyed. For instance, when the survey is done by phone, the result must be biased toward those who are well-off enough to have landline phones (or mobile phones these days).


## Caveats and Recommendations

Due to the focus of the project, the model has basic parameter settings. Also, the data used is almost 30 years old. While some might still hold, it is reasonable to doubt that the same would be applied to today. 