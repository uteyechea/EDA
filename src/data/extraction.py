import os
from pathlib import Path
import pandas as pd
import numpy as np

class Data():
    def __init__(self):
        # Define local working, data and libs directories
        self.root_dir = os.path.normpath(Path(__file__).resolve().parents[2])
        self.data_dir = os.path.normpath(os.path.join(self.root_dir, 'data'))
        self.raw = None  # The original, immutable data dump
    
    def extract(self, folder_name:str, file_name:str)->pd.DataFrame():
        assert type(folder_name) == str, f'{folder_name} expected dtype==str'
        assert type(file_name) == str, f'{file_name} expected dtype==str'
        data = pd.read_csv(os.path.join(self.data_dir, folder_name, file_name), na_values='n/a')
        self.raw = data
        return data.info()