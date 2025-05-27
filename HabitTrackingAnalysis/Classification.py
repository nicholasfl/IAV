from Day import Day
import pandas as pd
from typing import List


class Classification:
    def __init__(self, df: pd.DataFrame, lookback=-1):
        self.df = df
        self.day = len(df)
        if lookback == -1:
            self.lookback = self.day
        else:
            self.lookback = lookback

        self.day_split_dict = {}

        for i in range(self.lookback):
            current_day = self.get_day() - i
            i_day = Day(self.df, current_day)
            self.add_day_to_split_dict(i_day)
   
    def get_day(self):
        return self.day
 
    def get_df_len(self):
        return len(self.df)

    def add_day_to_split_dict(self, day_obj: Day):
        self.day_split_dict.update(
            {day_obj.get_day(): day_obj.get_splits()}
        )
        return

    def get_day_splits(self, day: int):
        if day == -1:
            day = self.get_df_len() - 1
        return self.day_split_dict.get(day, -1)

    def print_certain_day_splits(self, days: List[int]):
        for i in days:
            print(self.day_split_dict.get(i))
