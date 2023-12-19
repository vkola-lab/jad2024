# TAU SUVR Graphical Modeling Project 
In this repository, you will find all the code nessescary to:  
1. Clean and harmonize raw data csvs for ADNI and A4 cohorts 
2. Run Graphical LASSO model on bootstrapped Tau SUVR values 
3. Analyze graph level metrics to determine differences between global amyloid quartile groups 

## Graphical Model Pipeline 

### I. Data Cleaning Steps 
1. The raw csv files downloaded from ADNI and A4 are located in **mci_mri_graph/data_paths_and_cleaning/data/raw_csv_data**
   A. ADNI raw csv data will be in **mci_mri_graph/data_paths_and_cleaning/data/raw_csv_data/adni** <br>
   The raw csv with centiloid values is in this folder and named **UCBERKLEY_AMY_6MM_05Oct2023.csv** <br>
   The raw csv with tau suvr values is in this folder and named **hippo_UCBERKLEY_TAUPVC_6MM_13Nov2023.csv** <br>
   
   B. A4 data will be in **mci_mri_graph/data_paths_and_cleaning/data/raw_csv_data/a4** <br>
    The raw csv with centiloid values is in this folder and named **A4_PETSUVR_15_Aug2023.csv** <br>
    The raw csv with tau suvr values is in this folder and named **TAUSUVR_15_Aug2023.csv** <br>
   

2. Data cleaning scrips are located in **mci_mri_graph/data_paths_and_cleaning/data_cleaning_scrips** <br>
   In this folder you will find: <br>
   **merging_cent_tau_csvs.ipynb** which merges the centiloid and tau SUVR raw cvs into a master csv used for analysis and also uses a centiloid cut off value of >=21 to create a new csv with only amyloid positive patients with naming style **merged_adni/a4_at_amy_pos.csv** where adni/a4 is whichever dataset that csv belongs to <br>
   **adni_a4_data_harmonization.ipynb** which narrows down a list of 44 brain region shared across the adni and a4 data <br>
   **creating_quartiles** which creates centiloid quartile groups for adni and a4 and saves them to separate csv files that can be found in **mci_mri_graph/data_paths_and_cleaning/data/final_cleaned_quartiles**

#### Final Data Paths for CSVs with Tau SUVR Values for Each Centiloid Quartile Group
The file paths to the input data for the graphical models can be found in **mci_mri_graph/data_paths_and_cleaning/data/final_cleaned_quartiles** :  <br>
The ADNI data csv for each centiloid quartile will be in: **mci_mri_graph/Data_paths_and_cleaning/Data/ADNI**   <br>
The A4 data csv for each centiloid quartile will be in:  **mci_mri_graph/Data_paths_and_cleaning/Data/A4**  <br>
## II. Construct and Analyze Graphs 

### III. Graphical Modeling Scripts 
In this folder you will find scripts to created construct graphical models on the quartile data can be found in the folder **/mci_mri_graph/pet_graphs/current_tau_graphs** <br> 

**mci_mri_graph/pet_graphs/current_tau_graphs/bic.ipynb** is a script to show how different alpha values affect the sparsity of the precision matrix and BIC of the graphical model 

**mci_mri_graph/pet_graphs/current_tau_graphs/cent_pop_ggm_bootstrap.ipynb** is a script that takes the final quartile data for both ADNI and A4 and has the outline of code to create X number of bootstrap samples of the data and fit the graphical model to each bootstrapped sample and produce a graph visualization of all of these fitted bootstrapped models (calculated as the average precision matrix across all models). It also contains commented out skeletin code for calculating graph level metrics for each quartile group and then doing a t test to compare across groups. <br>

**mci_mri_graph/pet_graphs/current_tau_graphs/cent_pop_ggm_bootstrap.ipynb** is a similar to the previous script, except it also includes additionaly visualizations of node neighborhoods (nodes with more than one degree connectivity) <br>




