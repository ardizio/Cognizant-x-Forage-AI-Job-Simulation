# pip install pandas
# pip install scikit-learn
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate


TEST_SIZE = 0.2
K_FOLDS = 5


def load_data(path: str='dataframe.csv'):
    """Load the CSV file into a Pandas DataFrame

    Args:
        path (str): Path to csv file. Defaults to 'Data/sales.csv'.

    Returns:
        pd.DataFrame: Pandas dataframe object

    Raises:
        FileNotFoundError: If the file is not found.
    """
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Error loading dataset. '{path}' was not found.")


def load_model(model_path: str='model.pkl'):
    """
    Load a pickled machine learning model from the specified path using pandas.

    Args:
        model_path (str): The file path to the pickled model.

    Returns:
        object: The loaded machine learning model.
    """
    try:
        loaded_model = pd.read_pickle(model_path)
        return loaded_model
    except Exception as e:
        raise Exception(f"Error loading the model from {model_path}: {str(e)}")
    

def split_features_and_target(data: pd.DataFrame = None, target: str = 'estimated_stock_pct'):
    """Splits a DataFrame into features, X, and target variable, y.

    Args:
        data (pd.DataFrame, optional): Pandas DataFrame object. Defaults to None.
        target (str, optional): Target variable to isolate. Defaults to 'estimated_stock_pct'.

    Returns:
        X (pd.DataFrame): Pandas DataFrame features.
        y (pd.Series): Pandas Series target variable.
    """
    if data is None:
        raise ValueError("The 'data' parameter cannot be None.")

    X = data.drop(target, axis=1)
    y = data[target]
    return X, y


def evaluate_ml_model(X_train: pd.DataFrame, y_train: pd.Series, model, to_normalize=None, to_one_hot=None, cv=K_FOLDS):
    """
    Perform cross-validation with a machine learning model.

    Args:
        X_train (pd.DataFrame): Features for training.
        y_train (pd.Series): Target variable for training.
        model: The machine learning model to evaluate.
        to_normalize (list, optional): Columns to normalize. Defaults to None.
        to_one_hot (list, optional): Columns to one-hot encode. Defaults to None.
        cv (int, optional): Number of cross-validation folds. Defaults to 5.

    Returns:
        dict: Results including model name, cross-validation score, and evaluation metrics.
    """

    # Default columns for normalization and one-hot encoding
    if to_normalize is None:
        to_normalize = ['quantity', 'temperature', 'unit_price', 'day', 'weekday', 'hour']
    if to_one_hot is None:
        to_one_hot = ['category']

    # Define scorers for regression
    scorers = {
        'regression': {
            'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
            'MAPE': make_scorer(mean_absolute_percentage_error, greater_is_better=False),
            'RMSLE': make_scorer(mean_squared_log_error, greater_is_better=False),
            'R^2': make_scorer(r2_score)
        }
    }

    # Create a ColumnTransformer for preprocessing
    ct = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(), to_one_hot),
            ('normalizer', Normalizer(), to_normalize)
        ]
    )

    # Create a pipeline with preprocessing and the model
    pipe = Pipeline([
        ('ct', ct),
        ('model', model)
    ])

    # Initialize results dictionary
    results = {
        'ModelName': model.__class__.__name__,
        'CrossValidation': cv
    }

    for scorer_name, scorer in scorers['regression'].items():
        cv_results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scorer, error_score='raise')
        results[scorer_name] = cv_results['test_score'].mean()

    return results

def main():
    """Pipeline for loading data, model, splitting data, and running K-fold cross-validation.
    Prints the average cross-validation scores for both training and test sets.
    """
    # Load the data
    data = load_data()

    # Split data into features and target
    X, y = split_features_and_target(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE)

    # Load the machine learning model
    model = load_model()

    # Evaluate the model on the training set
    evaluated_model_train = evaluate_ml_model(X_train, y_train, model)
    print("Evaluation on Training Set:")
    print(evaluated_model_train)

    # Evaluate the model on the test set
    evaluated_model_test = evaluate_ml_model(X_test, y_test, model)
    print("Evaluation on Test Set:")
    print(evaluated_model_test)


