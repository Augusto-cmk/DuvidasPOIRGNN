import pandas as pd

class FileLoader:

    def __init__(self):
        pass

    def save_df_to_csv(self, df, filename, mode='w', header=True):
        #filename = DataSources.FILES_DIRECTORY.get_value() + filename
        df.to_csv(filename, index_label=False, index=False, mode=mode, header=header)

    def save_df_to_json(self, df, filename):
        #filename = DataSources.FILES_DIRECTORY.get_value() + filename
        filename = filename.replace(".csv", ".json")
        df.to_json(filename)