import json

def get_data_source_path():
    """
    Load the path of the directory that the data_storage should be sourced from.
    This is saved in the "data_source_path.txt" file

    Returns: The path as a string.
    """
    with open("local_util/settings/data_source_path.txt", 'r') as file:
        return file.read()


class InvalidOptionsFileError(Exception):
    """Raised when the options file does not contain a valid option."""
    pass


def _load_from_options_file(file_path):
    """
    Select the option from the source file that has a '/' as the first character.
    If none are selected with a '/', then return the first non-empty option from the file.

    Returns: The option selected.
    """
    with open(file_path, 'r') as file:
        lines = file.read().split('\n')
        if len(lines) > 0:
            for line in lines:

                if len(line) > 1:
                    if line[0] == '/':
                        return line[1:]

            for line in lines:
                if len(line) > 0:
                    return line

        raise InvalidOptionsFileError(f"There are no non-empty options in file: {file_path}.")


def get_data_source_type():
    """
    Select the option from the "run-type_options.txt" file that has a '/' as the first character.
    If none are selected with a '/', then return the first option from the file.

    Returns: The run-type selected.
    """
    return _load_from_options_file("local_util/settings/run-type_options.txt")


def get_reduction_type():
    """
    Select the option from the "reduction-type_options.txt" file that has a '/' as the first character.
    If none are selected with a '/', then return the first option from the file.

    Returns: The reduction-type selected.
    """
    return _load_from_options_file("local_util/settings/reduction-type_options.txt")


def get_program_arguments():
    with open("local_util/settings/program_arguments.json", 'r') as file:
        return json.load(file)
