from typing import TypeVar, List
import os

def create_training_folder(base_path, model_name):
    os.makedirs(base_path, exist_ok=True)
    directories = [name.split("_")[-1] for name in os.listdir(base_path) if name.startswith(f"{model_name}")]
    latest_i = -1

    for directory in directories:
        try:
            if int(directory) > latest_i:
                latest_i = int(directory)
        except Exception:  # ignore dir out of pattern
            continue

    path = os.path.join(base_path, f"{model_name}_{latest_i + 1}")
    os.mkdir(path)
    return path


def clear_folder(path):
    if os.path.exists(path):
        files = os.listdir(path)
        for file_name in files:
            os.remove(os.path.join(path, file_name))
    else:
        os.makedirs(path)


T = TypeVar('T')  # Declare a type variable

def where(array: List[T], condition) -> List[T]:
    return [e for e in array if condition(e)]

def single_where(array: List[T], condition) -> List[T]:
    for e in array:
        if condition(e):
            return e
    raise "Bad Element"
