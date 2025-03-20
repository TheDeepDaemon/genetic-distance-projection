import os

def get_numbered_unique_fpath(fpath):
    """
    If the filepath exists append a number to it to make it unique.
    For example: "example.txt" -> "example_1.txt"
    Always uses a number that hasn't been used yet.

    Args:
        fpath: The basic filepath to use. Example: "example.txt"

    Returns:
        The filepath with a number at the end. Example: "example_1.txt"
    """

    if os.path.exists(fpath):

        base_fpath, f_ext = os.path.splitext(fpath)

        count = 0

        while os.path.exists(fpath):
            count += 1
            fpath = f"{base_fpath}_{count}{f_ext}"

    return fpath