import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from Classification import Classification


def map_to_binary(value):
    try:
        float_value = float(value)
        return 1 if 0 <= float_value <= 10 and float_value > 5 else 0
    except (ValueError, TypeError):
        return value


def check_and_convert_date(date_str):
    try:
        return pd.to_datetime(date_str, format='%d.%m.%Y')
    except ValueError:
        try:
            return pd.to_datetime(date_str, format='%d-%m-%Y')
        except ValueError:
            return None


def save_dataframe_to_csv(df, output_name):
    """
    Saves the DataFrame to a CSV file with the given output name.

    Parameters:
        df (pd.DataFrame): DataFrame to be saved.
        output_name (str): Name of the output CSV file.
    """
    df.to_csv(output_name, index=False)


def preprocess_df(df: pd.DataFrame, binary):
    df.columns = df.iloc[0]
    new_df = df.iloc[1:]
    new_df = new_df.replace({"Y": 1, "N": 0})
    new_df = new_df.drop(["Notes"], axis=1, errors='ignore')
    new_df = new_df.dropna()
    to_bool_columns = ['Broken Sleep? (bool)', 'Breakfast? (bool)', 'Morning Routine? (bool)', 'Used ToDo List? (bool)', 'Nap? (bool)', 'Read? (bool)', 'Gym? (bool)', 'Fast Food? (bool)']
    for col in to_bool_columns:
        new_df[col] = new_df[col].astype('bool')

    new_df['Date'] = new_df['Date'].apply(check_and_convert_date)

    for col in new_df.columns:
        if new_df[col].dtype == 'object':
            if col != 'Day of Week':
                new_df[col] = new_df[col].astype(float)

    ohe = pd.get_dummies(new_df['Day of Week'], prefix='DayOfWeek')
    save_dataframe_to_csv(new_df, "before.csv")
    new_df = pd.concat([new_df, ohe], axis=1)
    save_dataframe_to_csv(new_df, "after.csv")
    new_df = new_df.drop('Day of Week', axis=1, errors='ignore')
    new_df = new_df.drop('Date', axis=1, errors='ignore')
    if binary:
        new_df['End of Day Productivity (0-10)'] = new_df['End of Day Productivity (0-10)'].apply(map_to_binary).astype(float)
    return new_df


def train_and_predict_input_classifier(classifier, X_train, y_train, X_test, y_test):
    start = time.time()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='binary')
    end = time.time()
    print(f"{classifier} took {end-start} time")
    return f1


def train_and_predict_input_regressor(regressor, X_train, y_train, X_test, y_test):
    start = time.time()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    end = time.time()
    print(f"{regressor} took {end-start} time")
    return mse


def plot_mse_with_varying_dataset_size(class_obj: Classification, lookback_window: int, ml_func, RANDOM_STATE, regression: bool):
    metrics = []
    for i in range(lookback_window):
        current_day = class_obj.get_day() - i
        day_splits = class_obj.get_day_splits(current_day)
        X_tr, X_te, y_tr, y_te = day_splits

        if regression:
            lookback_metric = train_and_predict_input_regressor(ml_func, X_tr, y_tr, X_te, y_te)
        else:
            lookback_metric = train_and_predict_input_classifier(ml_func, X_tr, y_tr, X_te, y_te)
        metrics.append(lookback_metric)
    return metrics[::-1]


def plot_models_over_forecast(ml_funcs, class_obj: Classification, LOOKBACK_WINDOW, RANDOM_STATE, regression, COLOURS):
    for index, (name, ml_func) in enumerate(ml_funcs.items()):
        model_metric = plot_mse_with_varying_dataset_size(class_obj, LOOKBACK_WINDOW, ml_func, RANDOM_STATE, regression)
        X = np.arange(class_obj.get_df_len() - LOOKBACK_WINDOW, class_obj.get_df_len(), 1)
        best_fit_curve = np.polyfit(X, model_metric, 1)
        best_fit_function = np.poly1d(best_fit_curve)

        if regression:
            plt.plot(X, model_metric, label=f"Model: {name}, mx + b = {best_fit_curve}", color=COLOURS[index % len(COLOURS)])
        else:
            plt.plot(X, model_metric, label=f"Model: {name}, Last F1 = {model_metric[-1]:.3f}", color=COLOURS[index % len(COLOURS)])

        plt.plot(X, best_fit_function(X), ls=":", color=COLOURS[index % len(COLOURS)])
    plt.grid()
    plt.legend()
    plt.xlabel("Database Day Cutoff")
    plt.xlim(class_obj.get_day() - LOOKBACK_WINDOW, class_obj.get_day())
    if regression:
        plt.ylabel("Model MSE")
    else:
        plt.ylabel("Model F1")
    plt.show()
    plt.close()


def load_regressor_funcs(RANDOM_STATE: int):
    regressor_funcs = {
        "RF": RandomForestRegressor(random_state=RANDOM_STATE),
        "SVM": SVR(C=1.0, epsilon=0.2),
        "GBR": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "ELA": ElasticNet(random_state=RANDOM_STATE),
        }
    return regressor_funcs


def load_classifier_funcs(RANDOM_STATE: int):
    classifier_funcs = {
        "RF": RandomForestClassifier(random_state=RANDOM_STATE),
        "ADA": AdaBoostClassifier(random_state=RANDOM_STATE),
        "NB": GaussianNB(),
        "LinSVM": SVC(kernel="linear", C=0.025, random_state=RANDOM_STATE),
        "DT": DecisionTreeClassifier(random_state=RANDOM_STATE),
    }
    return classifier_funcs


def plot_correlations(df, save_file_name=None):
    """ Plot correlations of features with the target variable.
    """
    target_column = "End of Day Productivity (0-10)"
    correlation_matrix = df.corr()
    correlations_with_target = correlation_matrix[target_column].sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    bar_plot = sns.barplot(x=correlations_with_target.values, y=correlations_with_target.index, palette='viridis')
    
    for index, value in enumerate(correlations_with_target.values):
        bar_plot.text(value, index, f'{value:.2f}', ha='left', va='center', fontsize=10, color='black')

    plt.title(f'Correlations with Target Variable: {target_column}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Features')

    if save_file_name:
        plt.savefig(save_file_name)
    else:
        plt.show()


def plot_binary_model_explanations(binary_class_obj: Classification):
    X_tr, _, y_tr, _ = binary_class_obj.get_day_splits(-1)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_tr, y_tr)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_tr)
    shap.summary_plot(shap_values, X_tr, plot_type="bar")