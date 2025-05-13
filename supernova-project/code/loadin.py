import pandas as pd

def load_supernova_data(file_path):
    """
    Load supernova data from a file.
    Parameters:
    file_path (str): Path to the data file.
    Returns:
    pd.DataFrame: DataFrame containing the supernova data.
    """
    try:
        data = pd.read_csv(file_path, sep=r'\s+', names=["redshift", "distance modulus"])
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    
def load_cov_data(file_path):
    """
    Load covariance data from a file.
    Parameters:
    file_path (str): Path to the covariance data file.
    Returns:
    pd.DataFrame: DataFrame containing the covariance data.
    """
    try:
        data = pd.read_csv(file_path, sep=r'\s+', names=["covariance"])
        return data
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None