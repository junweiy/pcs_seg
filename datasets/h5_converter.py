import numpy as np
import nibabel as nib
import os
from collections import Counter
import h5py

# Convert nifti to h5 files

# Modify the script for your needs
path = '/home/jy406/PCS_data/'
h5path = '/home/jy406/PCS_data_h5/'
filenames = os.listdir(path)

BMN_filenames = sorted([i for i in filenames if i.startswith('BMN')])
SUB_filenames = sorted([i for i in filenames if i.startswith('Sub')])
BMN_ids = list(Counter([i[:10] for i in BMN_filenames]).keys())
SUB_ids = list(Counter([i[:7] for i in SUB_filenames]).keys())

# Modify the dimension from 182 x 218 x 182 to 192 x 224 x 192 to be divisible by 32
scan_data = np.zeros([len(BMN_ids + SUB_ids), 192, 224, 192])
mask_combined = np.zeros([len(BMN_ids + SUB_ids), 192, 224, 192])

i = 0
for bmn_id in BMN_ids:
    curr_scan_filename = [i for i in BMN_filenames if i.startswith(bmn_id) and i.endswith('_acpc.nii.gz')][0]
    hf = h5py.File(h5path + curr_scan_filename + '.h5', 'w')
    img = nib.load(path + curr_scan_filename)
    data = img.get_fdata()
    scan_data[i,:182,:218,:182] = data
    hf.create_dataset('raw', data=scan_data[i,:,:,:])
    img = nib.load(path + bmn_id + '_PCS_LH.nii.gz')
    data = img.get_fdata()
    mask_combined[i,:182,:218,:182] += data
    img = nib.load(path + bmn_id + '_PCS_RH.nii.gz')
    data = img.get_fdata()
    mask_combined[i,:182,:218,:182] += data
    hf.create_dataset('label', data=mask_combined[i,:,:,:])
    i += 1

# Similar to the previous loop
for sub_id in SUB_ids:
    curr_scan_filename = [i for i in SUB_filenames if i.startswith(sub_id) and i.endswith('_acpc.nii.gz')][0]
    hf = h5py.File(h5path + curr_scan_filename + '.h5', 'w')
    img = nib.load(path + curr_scan_filename)
    data = img.get_fdata()
    scan_data[i,:182,:218,:182] = data
    hf.create_dataset('raw', data=scan_data[i,:,:,:])
    img = nib.load(path + sub_id + '_cT1_acpc_PCS_LH.nii.gz')
    data = img.get_fdata()
    mask_combined[i,:182,:218,:182] += data
    img = nib.load(path + sub_id + '_cT1_acpc_PCS_RH.nii.gz')
    data = img.get_fdata()
    mask_combined[i,:182,:218,:182] += data
    hf.create_dataset('label', data=mask_combined[i,:,:,:])
    i += 1

# Now starts to compute centre coordinate and add to h5 file as one attribute

# Threshold used to limit the number of samples to compute the coordinate
threshold = 0
file_paths = os.listdir(h5path)

# Write the file names that have the computed centre coordinate to a text file
f = open('valid_scans_list.txt', 'w')

for file_path in file_paths:
    curr_h5_file_path = h5path + file_path
    curr_h5_file = h5py.File(curr_h5_file_path, 'r')
    # Count the number of PCS voxels
    ones_count = np.count_nonzero(np.array(curr_h5_file['label']) == 1)
    # Only compute and save the coordinate when above the threshold
    if ones_count > threshold:
        f.write(h5path + file_path + '\n')
        coors = np.where(np.array(curr_h5_file['label']) == 1)
        coors = np.sum(coors, axis=1) / ones_count
        curr_h5_file.create_dataset('coor', data=coors)
        print("writing " + h5path + file_path + ', coor: ' + str(coors))
    curr_h5_file.close()
f.close()
