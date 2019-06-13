import os,sys,glob
import numpy as np
import nibabel as nib
from scipy.spatial import ConvexHull
from scipy.spatial import distance
from scipy import ndimage

# Takes as input a nifti PCS label and outputs an approximate length, by adding
# the length between furthest-neighbor voxels in every noncontiguous part of the
# mask. Make sure that each input file is just one label and doesn't constitute
# both the left and right labels.
# Usage:
#    python pcs_length.py mylabel.nii.gz
# Output:
#    <float>
# Author: Matthew Leming, Cambridge University, 2019
# matthew.j.leming at gmail.com
#
# Note: in your PCS labels, make sure that pixdim is correct, or the output
# length will be in terms of voxels and not millimeters.

def voxels_to_points(blob):
	points = []
	for x in range(blob.shape[0]):
		for y in range(blob.shape[1]):
			for z in range(blob.shape[2]):
				if blob[x,y,z] == 1:
					points.append([x,y,z])
	return np.array(points)

def find_furthest_two_voxels(blob,hull=True):
	points = voxels_to_points(blob)
	if hull:
		points = points[ConvexHull(points,qhull_options='QJ').vertices,:]
	dists = distance.cdist(points, points, 'euclidean')
	h1,h2 = np.unravel_index(np.argmax(dists),dists.shape)
	return (points[h1],points[h2])

def get_total_dist(blob,pixdim = [1,1,1]):
	total_dist = 0
	# Find, count, and label islands in the binary label
	label, num_features = ndimage.label(nifti_data, np.ones((3,3,3)))
	# Go through each island, find its two furthest points, calculate the
	# distance between them, and add them to a total. Weights by the original
	# pixel dimensions; make sure these are maintained so that the final
	# distance is an accurate mm count.
	for i in range(num_features):
		l = label == (i+1)
		if np.sum(l) > 4:
			p1,p2 = find_furthest_two_voxels(l)
			total_dist += (np.sum(((p1 - p2) * pixdim)**2))**0.5
	return total_dist

