# genetic-distance-projection  
A visualization method for evolutionary algorithms.

The project is made up of two main parts:  
- The `genome_data` folder, which contains the `GenomeData` class.
	- This is designed such that a person should be able to use it independently in another project.
	- This contains the implementation of the genetic distance projection (GDP) visualization method.
	- This is further broken down into reduction and visualization.
- The local implementation. 
	- This involves the loading of genome data, the loading of program arguments, and the use of GDP. 
	- This can be treated as an example of how to use the `GenomeData` class.
	- This local implementation is designed with data from EXAMM (a neuroevolution algorithm) in mind. 

This program is run by running the three files in sequence:
1. `data_pre_loading.py`: This pre-loads the data from the `.json` files, to streamline further use of the `GenomeData` class. 
2. `reduction.py`: This performs dimensionality reduction on the genome data, and saves the results to be used for visualization. 
3. `visualization.py`: This takes the reduced data and creates the visualizations. 

The local program arguments are chosen or set in the "local_util/settings" folder.
- New program arguments can be added
	- By adding a new value to the "program_arguments.json" file. 
	- Or by adding a new file to the "local_util/settings" folder, where the keyword for the program argument will be the filename (without the extension), and the value is whichever line in that file has an asterisk (`*`) at the beginning of the line.
