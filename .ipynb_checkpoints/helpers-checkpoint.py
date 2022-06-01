import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# This file contains basic functionalities that will help us in this project

# Reads/imports the dataset
def import_dataset(path = "C:\\Users\\ptodoran\\BigData\\1 IE\\kidney_disease.csv", fileName = "kidney_disease.csv"):
    if fileName == "kidney_disease.csv":
        df = pd.read_csv(path)
        return df
    else:
        print("Invalid dataset name. Choose 'kidney_disease.csv'")
        return

# Replacing the unknown values
# Droppping the index column since it matches the indexing
def clean_data(dataframe):
    if 'id' in dataframe.columns:
        data = dataframe.drop(columns=['id'])

    # Removing unnoticeable spaces that columns names might have
    data.columns = data.columns.str.strip()

    # Cleanup spaces and "?" values in "object" data points
    is_str_cols = data.dtypes == object
    str_columns = data.columns[is_str_cols]
    data[str_columns] = data[str_columns].apply(lambda s: s.str.strip())
    data = data.replace("?", np.nan)

    # Convert columns to numerical values if possible
    data = data.apply(pd.to_numeric, errors='ignore')

    # Pick the "true" string columns again
    is_str_cols = data.dtypes == object
    str_columns = data.columns[is_str_cols]

    # Replace NAN data points (object columns) with most used value
    data[str_columns] = data[str_columns].fillna(data.mode().iloc[0])

    # Replace numerical data points with mean of column
    data = data.fillna(data.mean())
    return data


def normalize_data(data):
    # One-hot encoding
    encoded_data = pd.get_dummies(data, drop_first=True)

    # Normalization
    normalized_data = (encoded_data - encoded_data.mean()) / (encoded_data.std())

    return normalized_data


def preprocess_data(data, classif="classification"):
    # Clean data
    data = clean_data(data)

    # Remove ground-truth
    target = data[classif]
    target = pd.get_dummies(target, drop_first=True)
    data = data.drop(columns=classif)

    # Normalized data
    data = normalize_data(data)

    return data, target












def feature_selection(df, method, variance_threshold=0.95):
    if method == 'PCA':
        X, _ = df.values[:, :-1], df.values[:, :-1]
        pca = PCA(n_components=len(df.columns) - 1)
        pca.fit(X)
        y = np.cumsum(pca.explained_variance_ratio_)

        nb_features = len(df.columns) - sum(y >= variance_threshold)
        # print(f'Number of features selected : {nb_features}')
        print(f' ✓ Data dimension successfuly reduced')
        new_X = pca.fit_transform(X)
        data = pd.DataFrame(new_X[:, :nb_features])
        data[df.columns[-1]] = df.iloc[:, -1]  # indexing the new dataframe

    return data


def split_data(X, y, test_size):
    # Split the data into a training set and a validation set

    # training set for features and labels, validation set features and labels.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def get_model(name):
    """
    @author: Ismail EL HADRAMI
    Get one ML model from a selection of models
    Parameters
    ----------
        name : name of the model
        type : string
    Returns
    -------
        model : Selected Classification Model
        type: sklearn class
    """

    models = {'svm': {'model': SVC,
                      'parameters': {'kernel': ['linear', 'rbf', 'sigmoid', 'poly'],
                                     'C': [1, 10],
                                     'degree': [2, 3],
                                     'gamma': ['scale', 'auto']
                                     }
                      },
              'lr': {'model': LogisticRegression,
                     'parameters': {'C': [1, 10],
                                    'fit_intercept': [True, False],
                                    'intercept_scaling': [1, 10],
                                    }
                     },
              'sgd_clf': {'model': SGDClassifier,
                          'parameters': {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
                                         'penalty': ['l1', 'l2'],
                                         'fit_intercept': [True, False],
                                         }
                          },
              'ab_clf': {'model': AdaBoostClassifier,
                         'parameters': {'n_estimators': [50, 100, 150],
                                        'algorithm': ['SAMME', 'SAMME.R'],
                                        'learning_rate': [0.1, 0.5, 1]
                                        }
                         },
              'rf_clf': {'model': RandomForestClassifier,
                         'parameters': {'n_estimators': [50, 100, 150],
                                        'criterion': ['gini', 'entropy'],
                                        'max_depth': [2, 5, 10, None],
                                        'bootstrap': [True, False],
                                        }
                         }
              }
    return models[name]


def train_model(name, X_train, y_train):
    """
    @author: Mohamed EL BAHA, Sami RMILI
    Compute the best model
    Parameters
    ----------
        name : name of the model
        type : string

        X_train : data to train
        type : Dataframe

        y_train : ground-truth for training
        type : Dataframe
    Returns
    -------
        model : Selected Classification Model
        type: sklearn class
    """
    # Get the model dictionnary by name

    m = get_model(name)
    print('Selected Model : ', m['model']())

    # Specify model and parameters
    model, params = m['model'], m['parameters']

    # Use grid search to come up with the best model (tune parameters)
    grid = GridSearchCV(model(), params, cv=5)
    grid.fit(X_train, y_train)
    best_train_score = grid.best_score_
    best_train_param = grid.best_params_
    model = grid.best_estimator_

    return model, best_train_score