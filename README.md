# MCI MRI Feature Embedding Generation & Graphical Model Repository 
In this repository, you will find all the code nessescary to:
1. Process regionally segmented MRI images into individual regional .nii image files 
2. Generate feature embeddings of regional .nii images 
3. Construct a gaussian graphical model of those regions for both progressive and stable MCI populations 
4. Analyze graph level metrics to determine differences between pMCI and sMCI groups 

## I. MRI Processing Pipeline 
In the folder mri_processing you will find all MRI processing code. 
A. This code will provide all the steps nessesary to take MRI images which has been aligned to the Neuromorphometrics regional atlas and generate 142 individual .nii files of each brain region. These files represent a binary mask of that brain region. 

![Screenshot 2023-05-09 at 2 08 00 PM](https://github.com/vkola-lab/mci_mri_graph/assets/90205073/891528e6-c94f-4c56-9717-bfac9d58bf16)

B. After generating the individual .nii binary masks of each region. You will need to run bin_to_sig_mask.ipynb to obtain the MRI signal intensity for that region. This code multiplies the binary mask but the original MRI image. Note: double check that the original MRI is alinged to the MNI space before running this. You can do so by printing the header information of the MRI and verifying qform: aligned. 


## II. Generate Feature Embeddings of Regional Image Segments 
In the folder generate_embeddings you will find all code required for generating a feature embedding for each .nii brain region image using an encoder neural network. 



## III. Construct Gaussian Graphical Model for pMCI and sMCI populations 
In the folder construct_ggm, you will find the code required to construct a gaussian graphical model of pMCI and sMCI populations, where each node represents a (1 region x n patient embeddings) vector containing the embeddings for that brain region across all patients within that population. 

## IV. Analyze Graph Level Metrics 
In the folder analyze_graphs you will find the code required to calculate graph level metrics such as modularity, effiency, etc. for both the pMCI and sMCI graphs and analyze differences between populations. 


