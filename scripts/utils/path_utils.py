from pathlib import Path

from scripts.setup.constants import C_NAME


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


def get_competition_data_path():
    data_path = get_data_directory_path()
    return data_path / C_NAME

def get_data_path():
    data_path = get_data_directory_path()
    return data_path

def get_competition_data_path_string():
    data_path = get_data_directory_path()
    return str(data_path / C_NAME)