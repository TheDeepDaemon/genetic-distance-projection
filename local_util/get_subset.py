
def get_subset(dictionary, keys):
    return {k: v for k, v in dictionary.items() if k in keys}
