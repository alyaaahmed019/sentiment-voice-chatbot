import re
import string
import pandas as pd
from datasets import load_dataset

class TextProcessor:
    def __init__(self):
        # Config (can be customized later)
        self.label_map = {0: 'anger', 1: 'joy', 2: 'optimism', 3: 'sadness'}

    def load_data(self, split: str = "train") -> pd.DataFrame:
        """
        Load the TweetEval emotion dataset as a pandas DataFrame.
        
        Args:
            split (str): dataset split to load ("train", "test", or "validation")
        
        Returns:
            pd.DataFrame: DataFrame containing the dataset
        """
        ds = load_dataset("cardiffnlp/tweet_eval", "emotion")
        return ds[split].to_pandas()

    def clean_text(self, text: str) -> str:
        """Clean raw text by removing punctuation, numbers, URLs, etc."""
        text = text.lower()
        text = re.sub(r'\[.*\]', '', text)   # remove text in square brackets
        text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # remove punctuation
        text = re.sub(r'\w*\d\w*', '', text)   # remove words containing numbers
        text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def count_labels(self, y: pd.Series) -> pd.Series:
        """Count frequency of labels in a pandas Series."""
        return y.value_counts()

    def convert_labels(self, y: int) -> str:
        """Convert numeric label to its text equivalent."""
        return self.label_map.get(y, 'unknown')

