# genetic-distance-projection  
A visualization method for evolutionary algorithms.

**FYI: This repo is regularly updated, so this README file is currently outdated. I am leaving this note just in case someone looks at this code before I get around to updating it.**

### Project Hierarchy
The project is made up of two main parts:
- The `gdp` folder, which contains the `GenomeData` class, and the `GenomeVisualizer` class.
	- This is designed such that a person should be able to use it independently in another project.
	- This contains the implementation of the genetic distance projection (GDP) visualization method.
	- Dimensionality reduction and visualization are handled separately.
- The local implementation. 
	- This involves the loading of genome data, the loading of program arguments, and the use of GDP. 
	- This can be treated as an example of how to use the `GenomeData` class. 
	- This local implementation is designed with data from EXAMM (a neuroevolution algorithm) in mind. 

### Running the Program
This program is run by running the three files in sequence:
1. `01_data_pre_loading.py`: This pre-loads the data from the `.json` files, to streamline further use of the `GenomeData` class. 
2. `02_reduction.py`: This performs dimensionality reduction on the genome data, and saves the results to be used for visualization. 
3. `03_visualization.py`: This takes the reduced data and creates the visualizations. 
4. `04_vis_archiver.py`: This moves everything from `vis_output` and moves it to `vis_archive`. It is useful if you want less clutter.

### Program Options and Configuration
The local program arguments are chosen or set in the `config` folder.  
- Program arguments can be set from the `config/config.yaml` file. If this doesn't exist, one will be automatically created by copying the `config/default_config.yaml` file. 
	- If any values in the `config.yaml` file are not filled in, the values in `default_config.yaml` will be used instead. 
- You will see two lists of arguments:
	- `args`: These are simple key-value pairs, with the name of the argument and the value. You can simply change the value of the argument here. 
	- `multiple_choice_args`: These are arguments with a few possible valid choices. Each has an `option` attribute and a `selected` attribute. To select an option, set `selected` to `true`. If multiple options are selected, the program will use the first one that is set to `true`. 
- The `data_source_path` variable must be set either in the `config.yaml` file, or as a command line argument. If this variable is not defined, the program will not run. 
