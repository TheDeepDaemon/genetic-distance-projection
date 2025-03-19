import yaml
import warnings
import os


class ConfigPaths:
    config_dir = "config"
    config_fname = "config.yaml"
    default_config_fname = "default_config.yaml"


def _load_program_arguments_from_file(fpath):
    with open(fpath, "r", encoding="utf-8") as file:
        yaml_data = yaml.safe_load(file)

        args = yaml_data["args"]

        mc_args_yaml = yaml_data["multiple_choice_args"]
        for keyword, options in mc_args_yaml.items():
            selected_option = None
            for option in options:
                if option["selected"]:
                    selected_option = option["option"]
                    break
            args[keyword] = selected_option

        return args


def load_program_arguments(default_ok=False):
    config_fpath = os.path.join(ConfigPaths.config_dir, ConfigPaths.config_fname)
    default_config_fpath = os.path.join(ConfigPaths.config_dir, ConfigPaths.default_config_fname)

    if not os.path.exists(config_fpath):
        with open(default_config_fpath, "r", encoding="utf-8") as def_file:
            with open(config_fpath, "w", encoding="utf-8") as conf_file:
                conf_file.write(def_file.read())

        print("\nConfig file does not exist, created a new config file.\n")

    args = _load_program_arguments_from_file(config_fpath)
    default_args = _load_program_arguments_from_file(default_config_fpath)

    for k, v in args.items():
        if v is None:

            # get the default value
            default_val = default_args[k]

            if not default_ok:
                # show warning message
                warning_message = (f"\'{k}\' has not been set in \'{config_fpath}\', "
                                   f"using default config value: \'{default_val}\'.")
                warnings.warn(warning_message, UserWarning)

            # set to default value
            args[k] = default_val

    # don't use the placeholder path to load data
    if args["data_source_path"] == default_args["data_source_path"]:
        print("Data source path is not set.")
        args["data_source_path"] = None

    return args
