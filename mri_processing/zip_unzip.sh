
#NACC zip unzip 
#script to compress them
SEARCH_DIR="/data2/MRI_PET_DATA/graph/NACC/Morph/mri_atlas/morph/roi"
cd $SEARCH_DIR

# Use the find command to search for folders that match the pattern
for dir in $(find $SEARCH_DIR -maxdepth 1 -type d -name 'NACC[0-9][0-9][0-9][0-9][0-9][0-9]' | sort); do
  echo "zipping files in folder: $dir"
  # Unzip all .nii.gz files in the directory
  for file in $dir/*.nii; do
    gzip $file
  done
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




#script to compress them
SEARCH_DIR="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/roi"
cd $SEARCH_DIR

# Use the find command to search for folders that match the pattern
for dir in $(find $SEARCH_DIR -maxdepth 1 -type d -name 'wneuromorphometrics_[0-9][0-9][0-9][0-9]*' | sort); do
  echo "zipping files in folder: $dir"
  # Unzip all .nii.gz files in the directory
  for file in $dir/*.nii; do
    gzip $file
  done
done

#output will be .gz zipped version 