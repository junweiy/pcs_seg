import h5py
import numpy as np
import os

# Script used to compute the average and max area that PCS segments occupy
# Part of the data preparation process, has nothing to do with training and prediction

data_path = '/home/jy406/PCS_data_h5/'
file_paths = os.listdir(data_path)

count = 0
max_h = 0
max_w = 0
max_d = 0

avg_h = 0
avg_w = 0
avg_d = 0

for file_path in file_paths:
	f = h5py.File(data_path + file_path.strip(), 'r')
	label = np.array(f['label'])
	f.close()
	if len(np.unique(label)) == 1:
		continue
	h, w, d = label.shape
	h_l, h_r = 0, h - 1
	w_l, w_r = 0, w - 1
	d_l, d_r = 0, d - 1
	for i in range(h):
		if len(np.unique(label[i,:,:])) == 2:
			h_l = i
			break
		elif i == h - 1:
			print("h_l ERROR")
	for i in range(h - 1, 0, -1):
		if len(np.unique(label[i,:,:])) == 2:
			h_r = i
			break
		elif i == 0:
			print("h_r ERROR")
	for i in range(w):
		if len(np.unique(label[:,i,:])) == 2:
			w_l = i
			break
		elif i == w - 1:
			print("w_l ERROR")
	for i in range(w - 1, 0, -1):
		if len(np.unique(label[:,i,:])) == 2:
			w_r = i
			break
		elif i == 0:
			print("w_r ERROR")
	for i in range(d):
		if len(np.unique(label[:,:,i])) == 2:
			d_l = i
			break
		elif i == d - 1:
			print("d_l ERROR")
	for i in range(d - 1, 0, -1):
		if len(np.unique(label[:,:,i])) == 2:
			d_r = i
			break
		elif i == 0:
			print("d_r ERROR")
	max_h = max(max_h, h_r - h_l)
	max_w = max(max_w, w_r - w_l)
	max_d = max(max_d, d_r - d_l)
	avg_h += h_r - h_l
	avg_w += w_r - w_l
	avg_d += d_r - d_l
	count += 1

print("Avg total_h: ", avg_h / count)
print("Avg total_w: ", avg_w / count)
print("Avg total_d: ", avg_d / count) 
