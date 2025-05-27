import pandas as pd
from sklearn.model_selection import train_test_split


class Day:
    def __init__(self, df: pd.DataFrame, day: int):
        self.day = day
        self.df = df
        
        if self.day > 10:
            self.X_tr, self.X_te, self.y_tr, self.y_te = self.split_dataset_for_training()
        else:
            self.X_tr, self.X_te, self.y_tr, self.y_te = ([], [], [], [])

    def get_day(self):
        return self.day

    def get_df(self):
        return self.df
    
    def split_dataset_for_training(self):
        df = self.get_df().copy()
        df = df.head(self.get_day())
        target_variable = "End of Day Productivity (0-10)"
        RANDOM_STATE = 42
        y = df[target_variable]
        X = df.drop(target_variable, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE
        )

        return X_train, X_test, y_train, y_test

    def get_splits(self):
        return (self.X_tr, self.X_te, self.y_tr, self.y_te)
