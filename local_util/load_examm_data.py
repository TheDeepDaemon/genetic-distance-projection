import json


def load_data(data_filepath):
    """
    Load EXAMM data from a json file.

    Args:
        data_filepath: The filepath to get the data from.

    Returns:
        The loaded EXAMM data.
    """
    with open(data_filepath, 'r', encoding="utf-8") as f:
        data_dict = json.load(f)
        return {int(k): v for k, v in data_dict.items()}
