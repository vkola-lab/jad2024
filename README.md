# Global Amyloid Burden Enhances Network Efficiency of Tau Propogation in the Brain
<div style="text-align:center;">
    <img src="graphical_lasso_schematic-1.png" alt="Screenshot">
</div>



In this repository, you will find all the code nessescary to:  
1. Clean and harmonize regional tau PET SUVr values and Centiloid values for ADNI and A4 cohorts.  
2. Run graphical machine learning model to learn the strongest conditional dependencies between tau accumulation in different brain regions and prune weaker associations,  
3. Analyze graph level metrics to determine differences in efficiency and organization of tau deposition at varying amyloid burdens.   

## I. Data Cleaning  
A. The tau and amyloid PET from ADNI and A4 are located in **mci_mri_graph/data_paths_and_cleaning/data/raw_csv_data**
   1. ADNI raw csv data will be in **mci_mri_graph/data_paths_and_cleaning/data/raw_csv_data/adni** <br>
   The raw csv with centiloid values is in this folder and named **UCBERKLEY_AMY_6MM_05Oct2023.csv** <br>
   The raw csv with tau suvr values is in this folder and named **hippo_UCBERKLEY_TAUPVC_6MM_13Nov2023.csv** <br>
   
   2. A4 data will be in **mci_mri_graph/data_paths_and_cleaning/data/raw_csv_data/a4** <br>
    The raw csv with centiloid values is in this folder and named **A4_PETSUVR_15_Aug2023.csv** <br>
    The raw csv with tau suvr values is in this folder and named **TAUSUVR_15_Aug2023.csv** <br>

    This data can be optained from the [ADNI]https://adni.loni.usc.edu/![image](https://github.com/vkola-lab/mci_mri_graph/assets/90205073/69c1eb69-f200-4784-b0e3-21ad647503c9)


B. Data cleaning scrips are located in **mci_mri_graph/data_paths_and_cleaning/data_cleaning_scrips** <br>

   1. **merging_cent_tau_csvs.ipynb** which merges the centiloid and tau SUVR raw csvs into a master csv used for analysis and also uses a centiloid cut off value of >=21 to create a new csv with only amyloid positive patients with naming style **merged_adni/a4_at_amy_pos.csv** where adni/a4 is whichever dataset that csv belongs to <br>
   2. **adni_a4_data_harmonization.ipynb**
   which narrows down a list of 44 brain region shared across the adni and a4 data <br>
   4. **creating_quartiles** which creates centiloid quartile groups for adni and a4 and saves them to separate csv files that can be found in **mci_mri_graph/data_paths_and_cleaning/data/final_cleaned_quartiles**

C. Final Data Paths for CSVs with Tau SUVR Values for Each Centiloid Quartile Group <br>
The file paths to the input data for the graphical models can be found in **mci_mri_graph/data_paths_and_cleaning/data/final_cleaned_quartiles** :  <br>
   1. The ADNI data csv for each centiloid quartile will be in: **mci_mri_graph/Data_paths_and_cleaning/Data/ADNI**   <br>
   2. The A4 data csv for each centiloid quartile will be in:  **mci_mri_graph/Data_paths_and_cleaning/Data/A4**  <br>

## II. Construct and Analyze Graphs 

### Graphical Modeling Scripts 
In this folder you will find scripts to created construct graphical models on the quartile data can be found in the folder **/mci_mri_graph/pet_graphs/current_tau_graphs** <br> 
   1. **mci_mri_graph/pet_graphs/current_tau_graphs/bic.ipynb** is a script to show how different alpha values affect the sparsity of the precision matrix and BIC of the graphical model 

   2. **mci_mri_graph/pet_graphs/current_tau_graphs/cent_pop_ggm_bootstrap.ipynb** is a script that takes the final quartile data for both ADNI and A4 and has the outline of code to create X number of bootstrap samples of the data and fit the graphical model to each bootstrapped sample and produce a graph visualization of all of these fitted bootstrapped models (calculated as the average precision matrix across all models). It also contains commented out skeletin code for calculating graph level metrics for each quartile group and then doing a t test to compare across groups. <br>

   3. **mci_mri_graph/pet_graphs/current_tau_graphs/cent_pop_ggm_bootstrap_visualization.ipynb** is a similar to the previous script, except it also includes additionaly visualizations of node neighborhoods (nodes with more than one degree connectivity) <br>




