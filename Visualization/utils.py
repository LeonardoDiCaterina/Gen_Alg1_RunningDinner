import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats  # Import the correct stats module


def boxplot_csv(*csv_paths):

    data_list = []
    filename_list = []
    # Check if the CSV files exist

    # Iterate over each CSV file
    for i, csv_path in enumerate(csv_paths):

        # Read in csv file 
        df = pd.read_csv(csv_path, index_col = 0 )

        # get out all the run_columns
        run_columns = [col for col in df.columns if 'run' in col]

        # get out the data
        data = df[run_columns].tail(1).values.flatten()
        
        filename = os.path.basename(csv_path).split('.')[0]
        
        # append as a tuple and create a dataframe out of it. 
        
        data_list.append(data)
        # Create a boxplot

        filename_list.append(filename)

    plot_df = pd.DataFrame(data_list, columns=[filename_list])

    # --- 3. Plot with matplotlib ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Boxplot: each column in plot_df becomes one box
    ax.boxplot(plot_df.values, labels=plot_df.columns, notch=True)
    ax.set_title("Fitness Distribution Across GA Runs")
    ax.set_xlabel("Experiment / CSV file")
    ax.set_ylabel("Fitness")
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Rotate x-tick labels if they overlap
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    plt.show()




def plot_p_value(file_1: str, file_2: str):
    

    df1 = pd.read_csv(file_1, index_col=0)
    df2 = pd.read_csv(file_2,  index_col=0)

    # 2) Spalten identifizieren, die "run" im Namen haben
    run_cols_1 = [c for c in df1.columns if "run" in c.lower()]
    
    run_cols_2 = [c for c in df2.columns if "run" in c.lower()]

    # 3) Für jede Generation Test durchführen und p-Values sammeln
    generations = df1.index.astype(int)  # falls der Index numerisch ist
    p_values = []

    for gen in generations:
        vals1 = df1.loc[gen, run_cols_1].values
        vals2 = df2.loc[gen, run_cols_2].values
        
        # unabhängiger t-Test; paired=True für gepaarte Tests
        t_stat, p = stats.ttest_ind(vals1, vals2, equal_var=False)
        p_values.append(p)

    # 4) p-Values plotten
    plt.figure(figsize=(8,4))
    plt.plot(generations, p_values, marker="o", linestyle="-")
    plt.axhline(0.05, color="red", linestyle="--", label="α = 0.05")
    plt.ylim(0,1)
    plt.xlabel("Generation")
    plt.ylabel("p-Value")
    plt.title("Hypothesentest per Generation")
    plt.legend()
    plt.tight_layout()
    plt.show()


    return p_values
