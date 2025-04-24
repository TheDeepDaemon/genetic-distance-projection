import yaml
from pathlib import Path
from typing import Union
import os


class _Config_:

    _DEFAULT_KWARGS_: dict = None

    def __init__(self):
        pass


def set_config_defaults(source_yaml_file: Union[str, os.PathLike]=None):
    if source_yaml_file is None:
        source_yaml_file = Path(__file__).parent / "defaults.yaml"

    with open(source_yaml_file, "r", encoding="utf-8") as file:
        _Config_._DEFAULT_KWARGS_ = yaml.safe_load(file)


def get_default_value(key):
    return _Config_._DEFAULT_KWARGS_[key]

def _set_kwargs_defaults_(kwargs: dict):
    for k, v in _Config_._DEFAULT_KWARGS_.items():
        kwargs.setdefault(k, v)
