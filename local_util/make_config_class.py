import yaml
from .load_config import load_program_arguments


def _infer_type(value):
    if isinstance(value, bool):
        return "bool"
    elif isinstance(value, int):
        return "int"
    elif isinstance(value, float):
        return "float"
    elif isinstance(value, str):
        return "str"
    elif value is None:
        return "Optional[Any]"
    else:
        return "Any"


def make_config_class():

    args = load_program_arguments()

    fields = []
    for k, v in args.items():
        _type = _infer_type(v)
        fields.append(f"    {k}: {_type}")

    class_def = \
        ("from local_util import load_program_arguments\n"
         "from dataclasses import dataclass\n"
         "from typing import Any, Optional\n"
         "\n"
         "@dataclass\n"
         "class ProgramArguments:\n"
         f"\n{'\n'.join(fields)}\n\n"
         "    def __init__(self, default_ok=False):\n"
         "        args = load_program_arguments(default_ok=default_ok)\n"
         "        for k, v in args.items():\n"
         "            setattr(self, k, v)\n"
         "        self.args = args\n"
         "\n"
         "    def get_subset(self, keys):\n"
         "        return {k: v for k, v in self.args.items() if k in keys}\n"
         "\n")

    with open("program_arguments.py", "w") as f:
        f.write(class_def)
