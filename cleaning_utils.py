import pandas as pd
import numpy as np
from typing import List, Dict

def sort_and_clean_df(df:pd.DataFrame, by:str='description') -> pd.DataFrame:
    df = df[df['description'].str.strip().astype(bool)]
    df.sort_values(by='description', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df



# print(df.head(5))