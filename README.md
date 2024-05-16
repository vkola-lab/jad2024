# Global Amyloid Burden Enhances Network Efficiency of Tau Propogation in the Brain
<div style="text-align:center;">
    <img src="graphical_lasso_schematic-1.png" alt="Screenshot">
</div>

<br></br>

In this repository, you will find all the code necessary to:  
1. Clean and harmonize regional tau PET SUVr values and Centiloid values for ADNI and A4 cohorts.  
2. Run graphical machine learning model to learn the strongest conditional dependencies between tau accumulation in different brain regions and prune weaker associations,  
3. Analyze graph level metrics to determine differences in efficiency and organization of tau deposition at varying amyloid burdens.   

## I. Data Cleaning  
Data can be obtained from [ADNI](https://adni.loni.usc.edu/data-samples/access-data/) and [A4](https://a4study.org/) websites upon request. 
   
B. Data cleaning scripts are located in **mci_mri_graph/data_paths_and_cleaning/data_cleaning_scrips** <br>

   1. **merging_cent_tau_csvs.ipynb** which merges the centiloid and tau SUVR raw csvs into a master csv used for analysis and also uses a centiloid cut off value of >=21 to create a new csv with only amyloid positive patients with naming style **merged_adni/a4_at_amy_pos.csv** where adni/a4 is whichever dataset that csv belongs to <br>
   2. **adni_a4_data_harmonization.ipynb**
   which narrows down a list of 44 brain region shared across the ADNI and A4 data <br>
   4. **creating_quartiles** which creates centiloid quartile groups for adni and a4 and saves them to separate csv files that can be found in **mci_mri_graph/data_paths_and_cleaning/data/final_cleaned_quartiles** 

C. Final Data Paths for CSVs with Tau SUVR Values for Each Centiloid Quartile Group <br>
The file paths to the input data for the graphical models can be found in **mci_mri_graph/data_paths_and_cleaning/data/final_cleaned_quartiles** :  <br>
   1. The ADNI data csv for each centiloid quartile will be in: **mci_mri_graph/Data_paths_and_cleaning/Data/ADNI**   <br>
   2. The A4 data csv for each centiloid quartile will be in:  **mci_mri_graph/Data_paths_and_cleaning/Data/A4**  <br>

## II. Construct and Analyze Graphs 

### Graphical Modeling Scripts 
In this folder you will find scripts to created construct graphical models on the quartile data can be found in the folder **/mci_mri_graph/pet_graphs/current_tau_graphs** <br> 
   1. **mci_mri_graph/pet_graphs/current_tau_graphs/bic.ipynb** is a script to show how different alpha values affect the sparsity of the precision matrix and BIC of the graphical model used to determine the optimal strength of the L1 regularization to apply to the model. A very high alpha results in a sparse precision matrix where almost all connections in the graph would be dropped and a very low alpha would result in no connections being dropped. Optimal alpha ensures that the model is learning the most important relationships by dropping weak or spurious relationshiops, while still retaining vital connections in the data.
    <div style="text-align:center;">
    <img src="nonzero_frac_bic-1.png" alt="Screenshot">
</div>
      
   3. 

   4. **mci_mri_graph/analyze_graphs/streamlined_graphs_centiloid_range.ipynb** is a script that creates 1000 bootstrap samples of the data and fits a probabilistic graphical model to each bootstrapped sample, produces graph visualizations of the model's learned tau graph structure, and calculates metrics like weighted clustering coefficient, average shortest path length, and weighted small world coefficient to analyze how tau efficiency increases at higher amyloid burdens. 
   5. **mci_mri_graph/analyze_graphs/sig_testing.ipynb** is a script that preforms significicance testing between mean graph metrics between amyloid groups. It preforms an ANOVA test for clustering coefficient and average shortest path length and a Kruskal-Wallis test on small world coefficient (appropriate significance tests were determined by running **mci_mri_graph/analyze_graphs/metrics_dis_checker.ipynb**)




