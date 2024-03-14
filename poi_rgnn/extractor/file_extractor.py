import json

import pandas as pd

class FileExtractor:

    def read_csv(self, filename):

        df = pd.read_csv(filename)
        if "Unnamed: 0" in list(df.columns):
            df.drop(columns=["Unnamed: 0"],inplace=True)

        return df

    def extract_ground_truth_from_csv(self, filename):
        df = pd.read_csv(filename)

        return df

    def read_json(self, filename):

        with open(filename) as file:
            data = json.loads(file.read())

        return data