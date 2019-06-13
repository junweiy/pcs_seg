from subprocess import call
import nibabel as nib
import numpy as np
import os

AFF = np.array([[  -1.,    0.,    0.,   90.],
               [   0.,    1.,    0., -126.],
               [   0.,    0.,    1.,  -72.],
               [   0.,    0.,    0.,    1.]])

def preprocess_nifti(nifti_path, output_path):
    img = nib.load(nifti_path)
    data = np.array(img.get_fdata())
    new_filename = nifti_path[:-7] + "_acpc.nii.gz"
    new_filename = output_path + new_filename.split('/')[-1]
    if data.shape == (192, 224, 192):
        return data
    call(["robustfov", "-i", nifti_path, "-m", "roi2full.mat", "-r", "input_robustfov.nii.gz"], stdout=open(os.devnull, 'wb'))
    call(["convert_xfm", "-omat", "full2roi.mat", "-inverse", "roi2full.mat"])
    call(["flirt", "-interp", "spline", "-in", "input_robustfov.nii.gz", "-ref", "./datasets/template", "-omat", "roi2std.mat", "-out", "acpc_mni.nii.gz"])
    call(["convert_xfm", "-omat", "full2std.mat", "-concat", "roi2std.mat", "full2roi.mat"])
    call(["aff2rigid", "full2std.mat", "outputmatrix"])
    call(["applywarp", "--rel", "--interp=spline", "-i", nifti_path, "-r", "./datasets/template", "--premat=outputmatrix", "-o", new_filename])
    call(["rm", "input_robustfov.nii.gz", "full2roi.mat", "full2std.mat", "outputmatrix", "roi2full.mat", "roi2std.mat", "acpc_mni.nii.gz"])
    data = save_new(new_filename)
    return np.array(data)

def save_new(new_filename):
    img = nib.load(new_filename)
    data = img.get_fdata()
    new_img = nib.Nifti1Image(data, AFF)
    nib.save(new_img, new_filename)
    return data