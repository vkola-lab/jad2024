# Increased Global Amyloid Burden Enhances Network Efficiency of Tau Propagation in the Brain

<div align="center">
    <img src="/readme_photos/new_model.png" alt="Screenshot">
</div>

<br />

In this repository, you will find all the code necessary to:  
1. Clean and harmonize regional tau PET SUVr and Centiloid values for ADNI and A4 cohorts.  
2. Run graphical LASSO machine learning model to estimate the strongest conditional dependencies between tau accumulation in different brain regions and prune weaker spurious correlations.  
3. Analyze graph metrics to determine differences in efficiency and organization of tau deposition at varying global amyloid burdens.   

## I. Data Cleaning  

A. Researchers can request the data used in this project from the [ADNI](https://adni.loni.usc.edu/data-samples/access-data/) and [A4](https://a4study.org/) websites.

<div align="center">
    <img src="/readme_photos/new_pop.png" alt="Screenshot">
</div>

B. Data cleaning scripts are located in [jad2024/data_paths_and_cleaning/data_cleaning_scripts](https://github.com/vkola-lab/jad2024/tree/main/data_paths_and_cleaning/data_cleaning_scripts)
<br />
- [merging_cent_tau_csvs.ipynb](https://github.com/vkola-lab/jad2024/blob/main/data_paths_and_cleaning/data_cleaning_scripts/merging_cent_tau_csvs.ipynb): merges the centiloid and tau SUVR raw csvs into a master csv used for analysis and applies a centiloid cut off value of >=21, established by [(Royse et al., 2021)](https://pubmed.ncbi.nlm.nih.gov/33971965/). The output is a new csv with only amyloid positive patients with naming style where adni/a4 is whichever dataset that csv belongs to.
- [adni_a4_data_harmonization.ipynb](https://github.com/vkola-lab/jad2024/blob/main/data_paths_and_cleaning/data_cleaning_scripts/adni_a4_data_harmonization.ipynb): creates new harmonized dataframes for ADNI and A4 with the tau SUVR values for 44 bilateral brain regions and saves them where a4/adni is the parent folder name for the csv depending on whichever cohort that data belongs to.
<br />
<div align="center">  
    <img src="/readme_photos/new_select.png" alt="Screenshot">
</div>

## II. Run Graphical Model, Visualize Graphs, and Analyze Metrics of Tau Efficiency

### Graphical Modeling Scripts 

In [jad2024/analyze_graphs](https://github.com/vkola-lab/jad2024/tree/main/analyze_graphs), you will find scripts for hyperparameter selection and running the graphical models on the data that has been divided into 3 centiloid quantile groups:

1. [hyperparameter_tuning/bic.ipynb](https://github.com/vkola-lab/jad2024/blob/main/analyze_graphs/hyperparameter_tuning/bic.ipynb): This script shows how different hyperparameter (alpha) values affect the sparsity of the precision and covariance matrices and BIC of the graphical model used to determine the optimal strength of the L1 regularization (alpha) that should be applied.

<div align="center">
    <img src="/readme_photos/new_bic.png" alt="Screenshot">
</div>

2. [construct_and_analyze_graphs/streamlined_graphs_allinone.ipynb](https://github.com/vkola-lab/jad2024/blob/main/analyze_graphs/construct_and_analyze_graphs/streamlined_graphs_allinone.ipynb): This script creates 1000 bootstrap samples of the data and fits a probabilistic graphical model to each bootstrapped sample, produces graph visualizations of the model's learned tau graph structure, and calculates metrics like weighted clustering coefficient, average shortest path length, and weighted small world coefficient to analyze how tau efficiency increases at higher amyloid burdens.

<div style="text-align:center;">
    <img src="/readme_photos/new_box.png" alt="Screenshot">
</div>

<div style="text-align:center;">
    <img src="/readme_photos/new_vis2.jpg" alt="Screenshot">
</div>

3. [construct_and_analyze_graphs/sig_testing.ipynb](https://github.com/vkola-lab/jad2024/blob/main/analyze_graphs/construct_and_analyze_graphs/sig_testing.ipynb): This script performs significance testing between mean graph metrics among amyloid groups. It performs an ANOVA test for clustering coefficient and average shortest path length and a Kruskal-Wallis test on small world coefficient.

<div align="center">
    <img src="/readme_photos/new_anova.png" alt="Screenshot">
</div>
