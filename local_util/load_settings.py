import json
import os.path


class SettingsPaths:
    settings_dir = "local_util/settings"
    program_arguments_fname = "program_arguments.json"


class InvalidOptionsFileError(Exception):
    """Raised when the options file does not contain a valid option."""
    pass


def _load_from_options_file(keyword):
    """
    Select the option from the source file that has a '*' as the first character.
    If none are selected with a '*', then return the first non-empty option from the file.

    Args:
        keyword: The keyword, which should be the same as the filename.

    Returns: The option selected.
    """
    file_path = os.path.join(SettingsPaths.settings_dir, keyword)

    with open(file_path, 'r') as file:
        lines = file.read().split('\n')
        if len(lines) > 0:
            for line in lines:

                if len(line) > 1:
                    if line[0] == '*':
                        return line[1:]

            for line in lines:
                if len(line) > 0:
                    return line

        raise InvalidOptionsFileError(f"There are no non-empty options in file: {file_path}.")


def _get_program_arguments_from_json():
    """
    Get only the program arguments defined in the json file.

    Returns: The program arguments from the json.
    """
    with open(os.path.join(SettingsPaths.settings_dir, SettingsPaths.program_arguments_fname), 'r') as file:
        return json.load(file)


def get_program_arguments():
    """
    Get the dictionary containing program arguments and their values.

    Returns: The arguments as a dictionary.
    """

    # this should return a dictionary
    args = _get_program_arguments_from_json()

    fnames = os.listdir(SettingsPaths.settings_dir)

    fnames.remove(SettingsPaths.program_arguments_fname)

    for fname in fnames:
        keyword = os.path.splitext(fname)[0]
        args[keyword] = _load_from_options_file(fname)

    return args
