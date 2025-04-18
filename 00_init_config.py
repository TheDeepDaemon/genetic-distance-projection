"""
This file is here to create the config file in config/config.yaml,
and to create the ProgramArguments class in program_arguments.py.

You only need to run this file once after cloning the repo,
but this file needs to be run before anything else.
"""
from local_util import make_config_class

if __name__=="__main__":

    # It saves to the gdp folder so the gdp code can access it.
    # This helps with IDE autocomplete features, which makes my life easier.
    make_config_class("gdp")
