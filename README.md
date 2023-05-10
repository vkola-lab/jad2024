# MCI MRI Feature Embedding Generation & Graphical Model Repository 
In this repository, you will find all the code nessescary to:
1. Process regionally segmented MRI images into individual regional .nii image files 
2. Generate feature embeddings of regional .nii images 
3. Construct a gaussian graphical model of those regions for both progressive and stable MCI populations 
4. Analyze graph level metrics to determine differences between pMCI and sMCI groups 

## I. MRI Processing Pipeline 
In the folder **mri_processing** you will find all MRI processing code. <br>
A. First run **fslmathsloop2.sh**. This script will take MRI images that has been aligned to the Neuromorphometrics regional atlas and generate 142 individual .nii files of each brain region. The resulting files represent a binary mask of that brain region. 

![Screenshot 2023-05-09 at 2 26 38 PM](https://github.com/vkola-lab/mci_mri_graph/assets/90205073/3c9227b9-5f42-4be0-a1ac-3e0d8c19ad70)

B. Next you will need to run **bin_to_sig_mask.ipynb** to obtain the MRI signal intensity for each region. This script multiplies the binary mask but the original MRI image. <br> Note: double check that the original MRI is alinged to the MNI space before running this. You can do so by printing the header information of the MRI and verifying qform: aligned. <br>
![Screenshot 2023-05-09 at 2 21 36 PM](https://github.com/vkola-lab/mci_mri_graph/assets/90205073/56ee2292-92d4-4620-af33-85bdbabad10a)

C. The resulting MRI region signal masks will likely take up a lot of storage space in their .nii format. You can run **zip_unzip.sh** in order to compress them into a .nii.gz format which will take up less space. Please note this script has blocks of code to both compress (gzip) the .nii files and unzip (gunzip) so just comment out the appropriate one depending on which action you want to perform upon the files.
## II. Generate Feature Embeddings of Regional Image Segments 
In the folder **generate_embeddings** you will find all code required for generating a feature embedding for each .nii brain region image using an encoder neural network. <br>
![Screenshot 2023-05-09 at 2 09 55 PM](https://github.com/vkola-lab/mci_mri_graph/assets/90205073/5edaf2c5-2330-47bc-b420-6396bf0c86c1)



## III. Construct Gaussian Graphical Model for pMCI and sMCI populations 
In the folder **construct_ggm**, you will find the code required to construct a gaussian graphical model of pMCI and sMCI populations, where each node represents a (1 region x n patient embeddings) vector containing the embeddings for that brain region across all patients within that population. <br>
![Screenshot 2023-05-09 at 2 24 10 PM](https://github.com/vkola-lab/mci_mri_graph/assets/90205073/35d09a14-2812-4b0c-88f6-5004f8406ba3)

## IV. Analyze Graph Level Metrics 
In the folder **analyze_graphs** you will find the code required to calculate graph level metrics such as modularity, effiency, etc. for both the pMCI and sMCI graphs and analyze differences between populations. 


