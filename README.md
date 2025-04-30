# genetic-distance-projection
The implementation of GDP, detailed in the GECCO-2025 paper titled "Visualizing the Dynamics of Neuroevolution with Genetic Distance Projections."    

The implementation of the visualization methodology is in the "gdp" folder. There are several examples of how to actually use it in the "examples" folder.     

To use it, you need to create a virtual environment (if you haven't already), and install the pip dependencies with the command:    
`pip install -r requirements.txt`

For the latest versions of the required packages, you can also just use:
`pip install --upgrade -r pip_imports.txt`

# Examples  
There are two examples listed in the examples folder designed to show people how to actually use the `gdp` module.     

## Example: From Previous EXAMM and NEAT Runs
This is an example of GDP being run on data collected from previous EXAMM and NEAT runs. There is a small data file in the `examm_data` directory that this can be run on.

Once that is done, you should be able to run this example without trouble. To do this, just activate the Python virtual environment and run `example1.py` or `example2.py`.   

## Example: Simple GA    
This is an example of how the `GenomeDataCollector` class can be used to collect data during a GA run. This does not require any data to be installed. To do this, activate the Python virtual environment and run `example.py`.

In this case, a simple genetic algorithm is solving the [knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem). 

There are eight different lists of items and values that determine the constraints of the knapsack problem here and they are stored in JSON files in the `knapsack_data` directory ("P01.json" through "P08.json"). 

If you would like to change the constraints so that the weights and values of the items are different, you can navigate to this part of the script:
```
if __name__=="__main__":  
    fname = "P01.json"  
    main(f"knapsack_data/{fname}")
```
and change the filename: "P01.json" to one of the others in the `knapsack_data` directory. 
