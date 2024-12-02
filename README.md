# Increased Global Amyloid Burden Enhances Network Efficiency of Tau Propagation in the Brain
<br></br>
<div style="text-align:center;">
    <img src="/readme_photos/figure2.jpg" alt="Screenshot">
</div>
<br></br>

In this repository, you will find all the code necessary to:  
1. Clean and harmonize regional tau PET SUVr values and Centiloid values for ADNI and A4 cohorts.  
2. Run graphical machine learning model to learn the strongest conditional dependencies between tau accumulation in different brain regions and prune weaker spurious correlations.  
3. Analyze graph metrics to determine differences in efficiency and organization of tau deposition at varying global amyloid burdens.   

## I. Data Cleaning  

A. Researchers can request the data used in this project from the [ADNI](https://adni.loni.usc.edu/data-samples/access-data/) and [A4](https://a4study.org/) websites. <br> 
1. A list of de-identified subject IDs of the ADNI participants used in this project are located in [jad2024/data_paths_and_cleaning/data/demographic_csvs/ADNI/ADNI_patient_ids.csv](https://github.com/vkola-lab/jad2024/blob/main/data_paths_and_cleaning/data/demographic_csvs/ADNI/ADNI_patient_ids.csv) <br> 
2. A list of de-identified subject IDs for the A4 participants used in this project are located in [jad2024/data_paths_and_cleaning/data/demographic_csvs/A4/A4_patient_ids.csv](https://github.com/vkola-lab/jad2024/blob/main/data_paths_and_cleaning/data/demographic_csvs/A4/A4_patient_ids.csv) <br>

    <img src="/readme_photos/selection.jpg" alt="Screenshot">
    <img src="/readme_photos/table1.png" alt="Screenshot">
</div>


B. Data cleaning scripts are located in [jad2024/data_paths_and_cleaning/data_cleaning_scripts](https://github.com/vkola-lab/jad2024/tree/main/data_paths_and_cleaning/data_cleaning_scripts) <br>

   1. [merging_cent_tau_csvs.ipynb](https://github.com/vkola-lab/jad2024/blob/main/data_paths_and_cleaning/data_cleaning_scripts/merging_cent_tau_csvs.ipynb) merges the centiloid and tau SUVR raw csvs into a master csv used for analysis and applies a centiloid cut off value of >=21 to create a new csv with only amyloid positive patients with naming style [merged_adni/adni_at_amy_pos.csv](https://github.com/vkola-lab/jad2024/blob/main/data_paths_and_cleaning/data/intermediate_data/adni/merged_adni_at_amy_pos.csv) where adni/a4 is whichever dataset that csv belongs to <br>
   2. [adni_a4_data_harmonization.ipynb](https://github.com/vkola-lab/jad2024/blob/main/data_paths_and_cleaning/data_cleaning_scripts/adni_a4_data_harmonization.ipynb) creates new harmonized dataframes for ADNI and A4 with the tau SUVR values for 44 bilateral brain regions and saves them to [jad2024/data_paths_and_cleaning/data/intermediate_data/a4/merged_a4_at_amy_pos_bi_harm.csv](https://github.com/vkola-lab/jad2024/blob/main/data_paths_and_cleaning/data/intermediate_data/a4/merged_a4_at_amy_pos_bi_harm.csv) where a4/adni is the parent folder name for the csv depending ion whichever cohort that data belongs to. <br>


## II. Run Graphical Model, Visualize Graphs, and Analyze Metrics of Tau Efficiency

### Graphical Modeling Scripts 
In [jad2024/analyze_graphs](https://github.com/vkola-lab/jad2024/tree/main/analyze_graphs) you will find scripts for hyperparamter selection and running the graphical models on the data that has been divided into 3 centiloid quantile groups <br> 
   1. [jad2024/analyze_graphs/hyperparamter_tuning/bic.ipynb](https://github.com/vkola-lab/jad2024/blob/main/analyze_graphs/hyperparameter_tuning/bic.ipynb) is a script to show how different hyperparameter (alpha) values affect the sparsity of the precision and covariance matrices and BIC of the graphical model used to determine the optimal strength of the L1 regularization (alpha) that should be applied. A very high alpha results in a sparse precision matrix where almost all connections in the graph would be dropped and a very low alpha would result in no connections being dropped. Choosing an optimal alpha value ensures that the model is learning the most important relationships by dropping weak or spurious relationshiops, while still retaining vital connections in the data.
    <div style="text-align:center;">
    <img src="/readme_photos/fig3_nonzero_frac.jpg" alt="Screenshot">
</div>


2. [jad2024/analyze_graphs/construct_and_analyze_graphs/streamlined_graphs_allinone](https://github.com/vkola-lab/jad2024/blob/main/analyze_graphs/construct_and_analyze_graphs/streamlined_graphs_allinone.ipynb) is a script that creates 1000 bootstrap samples of the data and fits a probabilistic graphical model to each bootstrapped sample, produces graph visualizations of the model's learned tau graph structure, and calculates metrics like weighted clustering coefficient, average shortest path length, and weighted small world coefficient to analyze how tau efficiency increases at higher amyloid burdens.

<div style="text-align:center;">
    <img src="/readme_photos/boxplot.jpg" alt="Screenshot">
    <img src="/readme_photos/table2.png" alt="Screenshot">
</div>


<div style="text-align:center;">
    <img src="/readme_photos/graph_vis.jpg" alt="Screenshot">
</div>


3. [jad2024/anaqlyze_graphs/construct_and_analyze_graphs/sig_testing.ipynb](https://github.com/vkola-lab/jad2024/blob/main/analyze_graphs/construct_and_analyze_graphs/sig_testing.ipynb)is a script that preforms significicance testing between mean graph metrics between amyloid groups. It preforms an ANOVA test for clustering coefficient and average shortest path length and a Kruskal-Wallis test on small world coefficient (significance tests were chosen by running [jad2024/anaqlyze_graphs/hyperparamter_tuning/metrics_dis_checker.ipynb](https://github.com/vkola-lab/jad2024/blob/main/analyze_graphs/hyperparameter_tuning/metrics_dis_checker.ipynb) to plot the distribution of each graph metric to determine the most approriate statistical test to apply to analyze differences across centiloid groups.)




