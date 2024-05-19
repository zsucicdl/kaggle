import ast
import os
import sys

from pathlib import Path
from stdlib_list import stdlib_list

from kaggle_scripts.setup.constants import C_NAME
from kaggle_scripts.setup.kaggle_api import KaggleApiBetter

api = KaggleApiBetter()
data_path = f'/data/input/{C_NAME}'


def extract_package_names(filename):
    with open(f'./{filename}', "r") as f:
        tree = ast.parse(f.read())
    packages = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                package_name = alias.name.split(".")[0]
                if package_name not in sys.modules:
                    packages.add(package_name)
        elif isinstance(node, ast.ImportFrom):
            package_name = node.module.split(".")[0]
            packages.add(package_name)
    return packages


def find_all_python_files(directory):
    python_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
            elif file.endswith(".ipynb"):
                py_file = os.path.join(root, file + ".py")
                os.system(f"jupyter nbconvert --to script {os.path.join(root, file)} --output {file} 2>/dev/null")
                python_files.append(py_file)
    return python_files


def generate_requirements(directory):
    all_packages = set()
    python_files = find_all_python_files(directory)
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    standard_libraries = set(stdlib_list(python_version))

    for file in python_files:
        try:
            packages = extract_package_names(file)
            all_packages |= packages
            if ".ipynb" in file:
                os.remove(file)
        except Exception as e:
            continue

    with open(f"{directory}/requirements.txt", "w") as f:
        for package in all_packages:
            if package not in standard_libraries:
                f.write(package + "\n")
    return all_packages


def convert_notebooks_to_python(directory):
    # iterate over files in directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".ipynb"):
                full_file_path = os.path.join(root, file)
                python_file_path = Path(full_file_path).with_suffix('.py')
                os.system(f"jupyter nbconvert --to script {full_file_path} --output {python_file_path} 2>/dev/null")


def generate_requirements2(directory):
    # convert notebooks to python files
    convert_notebooks_to_python(directory)

    # generate requirements.txt using pipreqs
    os.system(f'pipreqs {directory} --force')


# test
generate_requirements2(C_NAME)

all_packages = generate_requirements(C_NAME)
