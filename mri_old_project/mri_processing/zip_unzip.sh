
#NACC zip unzip 
#script to compress them
# SEARCH_DIR="/data2/MRI_PET_DATA/graph/NACC/Morph/mri_atlas/morph/roi"
# SEARCH_DIR="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas"
# cd $SEARCH_DIR

# # Use the find command to search for folders that match the pattern
# for dir in $(find $SEARCH_DIR -maxdepth 1 -type d -name 'NACC[0-9][0-9][0-9][0-9][0-9][0-9]' | sort); do
#   echo "zipping files in folder: $dir"
#   # Unzip all .nii.gz files in the directory
#   for file in $dir/*.nii; do
#   # -f force overwrite automatically yes command to overwrite file so it doesn't ask everytime 
#   #be really careful though and double check, only use on redundant files
#     gzip -f $file
#   done
# done

#!/bin/bash

# Directory containing the .nii files
dir="/data2/MRI_PET_DATA/graph/ADNI/ADNI_MRI_nii_recentered_cat12_cox_noqc"

# Use the find command to search for files that match the pattern
find "$dir" -name "*.nii" -type f | while read -r file; do
    echo "Zipping file: $file"
    gzip -f "$file"
done


#output will be .gz zipped version 
#unzip .nii output 

# # Set the directory to search
# SEARCH_DIR="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/roi"
# cd $SEARCH_DIR

# # Use the find command to search for folders that match the pattern
# for dir in $(find $SEARCH_DIR -maxdepth 1 -type d -name 'wneuromorphometrics_[0-9][0-9][0-9][0-9]*'); do
#   echo "Unzipping files in folder: $dir"
#   # Unzip all .nii.gz files in the directory
#   for file in $dir/*.nii.gz; do
#     gunzip $file
#   done
# done



#ADNI zip unzip 


#unzip .nii.gz output 

# # Set the directory to search
# SEARCH_DIR="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/roi"
# cd $SEARCH_DIR

# # Use the find command to search for folders that match the pattern
# for dir in $(find $SEARCH_DIR -maxdepth 1 -type d -name 'wneuromorphometrics_[0-9][0-9][0-9][0-9]*'); do
#   echo "Unzipping files in folder: $dir"
#   # Unzip all .nii.gz files in the directory
#   for file in $dir/*.nii.gz; do
#     gunzip $file
#   done
# done




# #script to compress them
# SEARCH_DIR="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/roi"
# cd $SEARCH_DIR

# # Use the find command to search for folders that match the pattern
# for dir in $(find $SEARCH_DIR -maxdepth 1 -type d -name 'wneuromorphometrics_[0-9][0-9][0-9][0-9]*' | sort); do
#   echo "zipping files in folder: $dir"
#   # Unzip all .nii.gz files in the directory
#   for file in $dir/*.nii; do
#     gzip $file
#   done
# done

#output will be .gz zipped version 
