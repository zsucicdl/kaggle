from pathlib import Path
import os


def get_current_working_dir():
    return Path.cwd()


def get_data_directory_path(sentinel=".git"):
    """
    Returns the path to the data folder located at the project root.

    Parameters:
        - sentinel: The name of a file or folder that exists at the project root.
                   Defaults to ".git".

    Returns:
        Path object pointing to the data folder.
    """
    # Start from the current working directory
    current_path = Path.cwd()

    # Traverse upwards until the sentinel is found or root is reached
    while current_path != current_path.parent:
        if (current_path / sentinel).exists():
            return current_path / "input"
        current_path = current_path.parent

    # If we're here, the sentinel was not found. Handle appropriately (e.g., raise an exception).
    raise FileNotFoundError(f"Project root with sentinel '{sentinel}' not found from starting directory {Path.cwd()}")


def get_project_root():
    return get_data_directory_path().parent


def get_competition_data_path(C_NAME):
    data_path = get_data_directory_path()
    return data_path / C_NAME

def get_data_path():
    data_path = get_data_directory_path()
    return data_path


def is_kaggle():
    """
    Checks if the code is running in a Kaggle environment.

    Returns:
    - (bool): True if running in Kaggle, else False.
    """
    return os.path.exists('/kaggle')


class KagglePaths:
    def __init__(self, c_name):
        if is_kaggle():
            prefix = '/kaggle/input'
        else:
            prefix = str(get_data_directory_path())
        self.dataset_data_path = os.path.join(prefix, "")
        self.competition_data_path = os.path.join(prefix, c_name)

    def get_dataset_data_path(self, dataset_name):
        return os.path.join(self.dataset_data_path, dataset_name)

    def get_competition_data_path(self, dataset_name):
        return os.path.join(self.competition_data_path, dataset_name)