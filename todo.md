# Left to do

- Run the grid search at least 30 times for each combination of parameters

- Save best fitness over generations for each run of combination of parameters 

- Create a function to get average fitness over generations for each combinaiton of parameters and append my csv files with avg

- Create a function to get std for both sides of the average and append the csv

- Plot the best 5-8 runs with average and stds 

- Test effect of mutations/crossovers

- Create main notebook file with a GA run with the best parameters and explain the best solution. Decode the genome - what is the optimized running dinner configuration?

- WRITE THE REPORT


- in grid search, figure out wich optimization problem is the most difficult to solve (and try increase the prob of mutation/xo towoards that)

- data ingestion show heatmap of the distance matrix (4 the report)

- clean up files


.
├── Classes
│   ├── Individual.py
│   ├── Solution.py
│   └── __pycache__
│       ├── Individual.cpython-312.pyc
│       └── Solution.cpython-312.pyc
├── Data
│   ├── coordinates_map_official.csv
│   └── distance_matrix_official.csv
├── Genetic_algorithm
│   ├──  __init__.py
│   ├── __pycache__
│   │   ├── base_individual.cpython-312.pyc
│   │   ├── config.cpython-312.pyc
│   │   ├── corssovers.cpython-312.pyc
│   │   ├── crossovers.cpython-312.pyc
│   │   ├── fitness.cpython-312.pyc
│   │   ├── genome.cpython-312.pyc
│   │   ├── mutations.cpython-312.pyc
│   │   ├── selection.cpython-312.pyc
│   │   ├── selection_alogrithms.cpython-312.pyc
│   │   └── solution_rd.cpython-312.pyc
│   ├── base_individual.py
│   ├── config.py
│   ├── crossovers.py
│   ├── fitness.py
│   ├── genome.py
│   ├── mutations.py
│   ├── selection.py
│   ├── selection_alogrithms.py
│   └── solution_rd.py
├── Notebooks
│   ├── Data_ingestion.ipynb
│   ├── GA_RUN.ipynb
│   ├── Individual_demo.ipynb
│   ├── SelMechDemo.ipynb
│   ├── Solution_RD_demo.ipynb
│   ├── Test.ipynb
│   └── solution_class.ipynb
├── Results
│   ├── binary_test.csv
│   ├── rank_selection.csv
│   └── tournament_selection.csv
└── Visualization
    └── utils.py