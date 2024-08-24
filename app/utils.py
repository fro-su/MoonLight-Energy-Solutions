import pandas as pd

def load_data(file):
    """Load data from a file-like object (e.g., uploaded CSV)."""
    return pd.read_csv(file)