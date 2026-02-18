import pandas as pd

def load_dataset(dataset_path):
    """
    Load a dataset from a given path and return it as a pandas DataFrame.

    Args:
        dataset_path (str): The path to the dataset file.

    Returns:
        pandas.DataFrame: A DataFrame containing the data from the dataset.
    """
    df = pd.read_csv(dataset_path)
    return df