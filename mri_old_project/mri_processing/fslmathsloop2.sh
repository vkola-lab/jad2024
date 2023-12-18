
#segments each neuromorphometrics atlas brain into 3d masks of each region

#add fslchfiletype to get the output files to be .nii not .nii.gz

#ADNI one works great
cd /data2/MRI_PET_DATA/graph/ADNI/mri_atlas/

#change this for NACC back to [0-9]
files=($(find . -maxdepth 1 -name 'wneuromorphometrics_[0-9][0-9][0-9][0-9]*' | sort -V))

for file in "${files[@]}"; do
  id=$(echo "$file" | grep -o 'wneuromorphometrics_[0-9][0-9][0-9][0-9]')
  DIR="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/"
  subj="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/${id}_mri.nii"
  echo "Processing subject ID: $id"

  mkdir -p "/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/roi/$id"
  echo "${subj}"
  #pwd 
  min_i=`fslstats "${subj}" -l 0.1 -R | cut -d ' ' -f1`
  max_i=`fslstats "${subj}" -l 0.1 -R | cut -d ' ' -f2`
  min_i=$(printf "%.0f" $min_i)
  max_i=$(printf "%.0f" $max_i)

  for i in `seq ${min_i} ${max_i}`; do
    output_dir="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/roi/${id}"
    echo "${output_dir}"
    filename=$(printf "%03d.nii" $i) 
    fslmaths "${subj}" -thr $i -uthr $i -bin -odt float "${output_dir}/${filename}"
    # echo ${id}
    echo ${filename}
  done 
done










#NACC 

# cd /data2/MRI_PET_DATA/graph/NACC/Morph/mri_atlas/morph

# #change this for NACC back to [0-9]
# files=$(find . -maxdepth 1 -name 'wneuromorphometrics_NACC[0-9][0-9][0-9][0-9][0-9][0-9]_mri.nii'
# )

# for file in "${files[@]}"; do
#   id=$(echo "$file" | grep -o 'NACC[0-9][0-9][0-9][0-9][0-9][0-9]_mri.nii')
# ')
#   DIR="/data2/MRI_PET_DATA/graph/NACC/Morph/mri_atlas/morph"
#   subj="/data2/MRI_PET_DATA/graph/NACC/Morph/mri_atlas/morph/${id}_mri.nii"
#   echo "Processing subject ID: $id"

#   mkdir -p "/data2/MRI_PET_DATA/graph/NACC/Morph/mri_atlas/morph/roi/$id"
#   echo "${subj}"
#   #pwd 
#   min_i=`fslstats "${subj}" -l 0.1 -R | cut -d ' ' -f1`
#   max_i=`fslstats "${subj}" -l 0.1 -R | cut -d ' ' -f2`
#   min_i=$(printf "%.0f" $min_i)
#   max_i=$(printf "%.0f" $max_i)

#   for i in `seq ${min_i} ${max_i}`; do
#     output_dir="/data2/MRI_PET_DATA/graph/NACC/Morph/mri_atlas/morph/roi/${id}"
#     echo "${output_dir}"
#     filename=$(printf "%03d.nii" $i) 
#     fslmaths "${subj}" -thr $i -uthr $i -bin "${output_dir}/${filename}"
#     # echo ${id}
#     echo ${filename}
#   done 
# done


# # cd /data2/MRI_PET_DATA/graph/ADNI/mri_atlas/

# # #change this for NACC back to [0-9]
# # files=($(find . -maxdepth 1 -name 'wneuromorphometrics_[0-9][0-9][0-9][0-9]*' | sort -V))

# # for file in "${files[@]}"; do
# #   id=$(echo "$file" | grep -o 'wneuromorphometrics_[0-9][0-9][0-9][0-9]')
# #   DIR="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/"
# #   subj="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/${id}_mri.nii"
# #   echo "Processing subject ID: $id"

# #   mkdir -p "/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/roi/$id"
# #   echo "${subj}"
# #   #pwd 
# #   min_i=`fslstats "${subj}" -l 0.1 -R | cut -d ' ' -f1`
# #   max_i=`fslstats "${subj}" -l 0.1 -R | cut -d ' ' -f2`
# #   min_i=$(printf "%.0f" $min_i)
# #   max_i=$(printf "%.0f" $max_i)

# #   for i in `seq ${min_i} ${max_i}`; do
# #     output_dir="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/roi/${id}"
# #     echo "${output_dir}"
# #     filename=$(printf "%03d.nii" $i) 
# #     fslmaths "${subj}" -thr $i -uthr $i -bin "${output_dir}/${filename}"
# #     # echo ${id}
# #     echo ${filename}
# #   done 
# # done
































# #revised fsl maths loop to generate 3d regional masks
# #it worked and generated 142 segments for subj 0173
# cd /data2/MRI_PET_DATA/graph/ADNI/mri_atlas/

# files=($(find . -maxdepth 1 -name 'wneuromorphometrics_[0-9][0-9][0-9][0-9]*'))

# for file in "${files[@]}"; do
#   id=$(echo "$file" | grep -o 'wneuromorphometrics_[0-9][0-9][0-9][0-9]')
#   mkdir -p "/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/roi/$id"
#   DIR="/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/"

#   for subj in $(find . -name '*.nii'); do
#       echo "Processing subject ID: $id"
#       min_i=`fslstats "${subj}" -l 0.1 -R | cut -d ' ' -f1`
#       max_i=`fslstats "${subj}" -l 0.1 -R | cut -d ' ' -f2`

#       min_i=$(printf "%.0f" $min_i)
#       max_i=$(printf "%.0f" $max_i)

#       for i in `seq ${min_i} ${max_i}`; do
#           file=$(printf "%03d.nii" $i) 
#           fslmaths "${subj}" -thr $i -uthr $i -bin \
#               "/data2/MRI_PET_DATA/graph/ADNI/mri_atlas/roi/$id/${file}"
#       done 
#   done
# done