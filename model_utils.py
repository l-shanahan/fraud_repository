import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def scale_feature_vector(df_customers):

    """
    Scales the customer feature data using standardization (z-score normalization),
    which rescales features to have a mean of 0 and a standard deviation of 1.

    Args:
    df_customers (pandas.DataFrame): DataFrame containing unscaled customer features.

    Returns:
    pandas.DataFrame: DataFrame with scaled customer features.
    """


    #normalising features
    scaler = StandardScaler()
    df_customers_scaled = pd.DataFrame(scaler.fit_transform(df_customers), columns=df_customers.columns)
    
    return df_customers_scaled


def separate_xy(df_customers_scaled):

    """
    Separates the features and target variable from the customer DataFrame.

    Args:
    df_customers_scaled (pandas.DataFrame): DataFrame containing scaled customer features.

    Returns:
    tuple: Contains two elements:
           X (pandas.DataFrame): Features DataFrame.
           y (pandas.Series): Target variable series indicating fraudulent status.
    """


    X = df_customers_scaled.drop(columns=['fraudulent']).astype(float) #features
    y = df_customers_scaled['fraudulent'].astype(int) #target

    return X, y

def tt_split(df_customers_scaled):

    """
    Splits customer data into training and test sets.

    Args:
    df_customers_scaled (pandas.DataFrame): DataFrame containing scaled customer features.

    Returns:
    tuple: Contains four elements:
           X_train (pandas.DataFrame): Training features.
           X_test (pandas.DataFrame): Test features.
           y_train (pandas.Series): Training target variable.
           y_test (pandas.Series): Test target variable.
    """


    X, y = separate_xy(df_customers_scaled)

    #splitting into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):

    """
    Trains a Random Forest Classifier on the provided training data.

    Args:
    X_train (pandas.DataFrame): Training features.
    y_train (pandas.Series): Training target variable.

    Returns:
    RandomForestClassifier: Trained model.
    """


    #initialising and training classifier
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    return classifier


def prediction_func(classifier, X):

    """
    Predicts the target variable using the trained classifier for the provided features.

    Args:
    classifier (RandomForestClassifier): Trained model.
    X (pandas.DataFrame): Features DataFrame for which predictions are to be made.

    Returns:
    numpy.ndarray: Predictions array.
    """


    y_pred = classifier.predict(X)

    return y_pred


def create_model(feature_df):

    """
    Creates a Random Forest Classifier model by scaling features, splitting the data,
    training the model, and then printing the model's accuracy on test data.

    Args:
    feature_df (pandas.DataFrame): DataFrame containing the customer features to be modeled.

    Returns:
    RandomForestClassifier: Trained classifier.
    """


    feature_df_scaled = scale_feature_vector(feature_df)
    X_train, X_test, y_train, y_test = tt_split(feature_df_scaled)
    classifier = train_model(X_train, y_train)
    y_pred = prediction_func(classifier, X_test)
    print('Model created with test accuracy of', accuracy_score(y_test, y_pred))

    return classifier

