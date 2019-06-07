import torch
import torch.utils.data as utils

import numpy as np

import h5py

import os

import sklearn.preprocessing

W, H, D = 63, 77, 93

# Load the dataset with cropped images as two data loaders
# The coordinate used for cropping is the PREDICTED centre
def crop_dataset_window(loaders, trainer):
	# Initialise variables to save processed dataset
	cropped_scans = []
	cropped_labels = []
	# Assign useful variables
	model = trainer.model
	device = trainer.device
	train_dl = generate_loader(model, device, loaders['train'])
	val_dl = generate_loader(model, device, loaders['val'])
	return {'train': train_dl, 'val': val_dl}

# Generate the data loader for cropped image
def generate_loader(model, device, loader):
	# Load original dataset
	scans, labels = process_iterator(loader)
	assert scans.shape == labels.shape, "Scan shape must be equal to label shape."
	model.eval()
	with torch.no_grad():
		for scan, label in zip(scans, labels):
			scan_tensor = scan.view((1,) + scan.shape).to(device)
			# Predict the coordinates of PCS centre
			x, y, z = model(scan_tensor).squeeze().cpu().numpy()
			# Generate slices with certain dim around the centre
			cropped_scans.append(generate_slice(scan, x, y, z))
			cropped_labels.append(generate_slice(label, x, y, z))

	return np_to_dataloader(cropped_scans, cropped_labels)

def process_iterator(iterator):
	xs = []
	ys = []
	for i in iterator:
		xs.append(i[0].squeeze().numpy())
		ys.append(i[1].squeeze().numpy())
	xs = np.array(xs)
	ys = np.array(ys)
	return xs, ys

# An alternative function to crop_dataset_window used to generate
# data loaders for training
# files contain a list of absolute paths to all h5 data files
def crop_data_window_training(files=None):
	# Load the list of filenames if not provided
	if not files:
		f = open('./datasets/valid_scans_list.txt', 'r')
		files = [i.strip() for i in f.readlines()]
		f.close()
	scans = []
	labels = []
	for file in files:
		h5_file = h5py.File(file, 'r')
		scan = np.array(h5_file['raw'])
		label = np.array(h5_file['label'])
		x, y, z = np.array(h5_file['coor'])
		cropped_scan = generate_slice(scan, x, y, z)
		cropped_label = generate_slice(label, x, y, z)
		scans.append(cropped_scan)
		labels.append(cropped_label)

	# Split the data into training and validation set
	split_index = int(len(scans) * 0.9)
	val_scans = np.array(scans[split_index:], dtype=np.float32)
	val_labels = np.array(labels[split_index:], dtype=np.long)
	scans = np.array(scans[:split_index], dtype=np.float32)
	labels = np.array(labels[:split_index], dtype=np.long)
	
	# Prepare the correct dimension
	n1, w, h, d = val_scans.shape
	n2 = scans.shape[0]
	scans = np.reshape(scans, (n2, 1, w, h, d))
	labels = np.reshape(labels, (n2, w, h, d))
	val_scans = np.reshape(val_scans, (n1, 1, w, h, d))
	val_labels = np.reshape(val_labels, (n1, w, h, d))
	
	return {'train': np_to_dataloader(scans, labels), 'val': np_to_dataloader(val_scans, val_labels)}

# Convert numpy array to pytorch dataloader
def np_to_dataloader(scans, labels):
	tensor_scans = torch.stack([torch.tensor(i, dtype=torch.float32) for i in scans])
	tensor_labels = torch.stack([torch.tensor(i, dtype=torch.long) for i in labels])

	dataset = utils.TensorDataset(tensor_scans, tensor_labels)
	dataloader = utils.DataLoader(dataset)
	return dataloader

# Generate image patch around the coordinate (x, y, z) with defined window size
def generate_slice(scan, x, y, z):
	if len(scan.shape) == 4:
		scan = np.reshape(scan, scan.shape[1:])
	x, y, z = int(x), int(y), int(z)
	w, h, d = scan.shape
	tmp = np.zeros((w + W, h + H, d + D))
	tmp[int(W/2):int(W/2+w),int(H/2):int(H/2+h),int(D/2):int(D/2+d)] = scan
	return tmp[int(x):int(x+W),int(y):int(y+H),int(z):int(z+D)]

# Recover the whole image with given shape from patch centred at (x, y, z)
# label has size (w, h, d)
def recover_patch(label, x, y, z, shape):
	tmp = np.zeros(shape)
	w, h, d = shape
	l_w, l_h, l_d = label.shape
	tmp[int(max(x-W/2,0)):int(min(x+W/2,w)),int(max(y-H/2,0)):int(min(y+H/2,h)),int(max(z-D/2,0)):int(min(z+D/2,d))] = label[int(max(W/2-x+1,0)):int(l_w-max(0,x+W/2-w-1)),int(max(H/2-y+1,0)):int(l_h-max(0,y+H/2-h-1)),int(max(D/2-z+1,0)):int(l_d-max(0,z+D/2-d-1))]
	return tmp

# 4D version of recover_path
# label has size (2, w, h, d), the first channel represents the probability of class 0/1
def recover_patch_4d(label, x, y, z, shape):
	x, y, z = int(x+1), int(y+1), int(z+1)
	tmp = np.zeros((1,) + tuple(shape))
	tmp[0,0,:,:,:] = 1
	w, h, d = shape[1:]
	l_w, l_h, l_d = label.shape[1:]
	tmp[0,:,int(max(x-W/2,0)):int(min(x+W/2,w)),int(max(y-H/2,0)):int(min(y+H/2,h)),int(max(z-D/2,0)):int(min(z+D/2,d))] = label[:,int(max(W/2-x+1,0)):int(l_w-max(0,x+W/2-w-1)),int(max(H/2-y+1,0)):int(l_h-max(0,y+H/2-h-1)),int(max(D/2-z+1,0)):int(l_d-max(0,z+D/2-d-1))]
	return tmp

