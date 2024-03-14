from pathlib import Path
from matplotlib import pyplot
import numpy as np
import pandas as pd

from loader.file_loader import FileLoader

class NextPoiCategoryPredictionSequencesGenerationLoader(FileLoader):

    def __init__(self):
        pass

    def sequences_to_csv(self, df, filename):

        self.save_df_to_csv(df, filename)