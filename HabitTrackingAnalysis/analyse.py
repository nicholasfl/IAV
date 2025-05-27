import pandas as pd
import requests
from io import StringIO
from Classification import Classification
from helper import load_classifier_funcs, load_regressor_funcs, plot_binary_model_explanations, plot_correlations, plot_models_over_forecast, preprocess_df


def main():
    print("Starting Script")

    COLOURS = ["blue", "red", "green", "brown", "orange", "purple", "gray"]
    RANDOM_STATE = 1
    LOOKBACK_WINDOW = 35

    nicholas_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRgMOuzHwnfnBfxk0FWnrGLNs3oEfQ1RGu1Jg7RaSbG5iDlQmR9lA5ufbRydzAUwxxb3ZwbNJh6_OJK/pub?output=csv"
    petra_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTyA3dO4pFccbafkXtBQLOrC08r4CSt6vWwa8uU8ElX9EHvEGCwiY_riLiRL2m7EfpxLCRqbC_AlkSO/pub?output=csv"
 
    try:
        nicholas_response = requests.get(nicholas_url)
        petra_response = requests.get(petra_url)
    except requests.exceptions.ConnectionError:
        print("Unable to access live file: loading backup")

    successful_retrieval = nicholas_response.status_code == 200 and petra_response.status_code == 200

    if not successful_retrieval:
        print("Datasets not successfully retrieved.")
        print("Response codes: {nicholas_response.status_code}, {petra_response.status_code}")
        most_recent_df = pd.read_csv("up_to_date.csv")
        most_recent_binary_df = pd.read_csv("up_to_date_binary.csv")

        df = most_recent_df
        binary_df = most_recent_binary_df

    else:
        print("All Datasets Loaded")

        nicholas_csv_data = StringIO(nicholas_response.text)
        nicholas_data = pd.read_csv(nicholas_csv_data)
        nicholas_data.to_csv("nicholas_data.csv")

        petra_csv_data = StringIO(petra_response.text)
        petra_data = pd.read_csv(petra_csv_data)
        petra_data.to_csv("petra_data.csv")

        nicholas_df = preprocess_df(nicholas_data, binary=False)
        petra_df = preprocess_df(petra_data, binary=False)

        nicholas_binary_df = preprocess_df(nicholas_data, binary=True)
        petra_binary_df = preprocess_df(petra_data, binary=True)
        
        
        #df = pd.concat([nicholas_df, petra_df], axis=0)
        #binary_df = pd.concat([nicholas_binary_df, petra_binary_df], axis=0)

        df = nicholas_df
        binary_df = nicholas_binary_df

        df.to_csv("up_to_date.csv")
        binary_df.to_csv("up_to_date_binary.csv")

    print(f"{df.shape}")
    print(f"{binary_df.shape}")

    class_obj = Classification(df)
    binary_class_obj = Classification(binary_df)

    plot_models_over_forecast(load_classifier_funcs(RANDOM_STATE), binary_class_obj, LOOKBACK_WINDOW, RANDOM_STATE, False, COLOURS)
    plot_models_over_forecast(load_regressor_funcs(RANDOM_STATE), class_obj, LOOKBACK_WINDOW, RANDOM_STATE, True, COLOURS)

    plot_binary_model_explanations(binary_class_obj)

    plot_correlations(df, "petra_explanations.png")


if __name__ == "__main__":
    main()
