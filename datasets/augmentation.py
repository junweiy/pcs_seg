import h5py
import numpy as np
import os

import argparse

from crop_window import generate_slice

from scipy.ndimage.filters import gaussian_filter

# Argument template
# output_dir = '/local/sdb/jy406/PCS_data_h5_1k/'
# input_dir = '/home/jy406/PCS_data_h5/'
# shifts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25]

def main():
	parser = argparse.ArgumentParser(description='Data Augmentation')
	parser.add_argument('--input-path', type=str, default=None,
	                        help='path to the dataset.')
	parser.add_argument('--output-path', required=True, type=str,
	                        help='path to save augmented dataset.')
	parser.add_argument('--interval', required=True, type=int,
	                        help='interval for augmentation.')
	args = parser.parse_args()

	# Load data paths
	if args.input_path:
		files = os.listdir(args.input_path)
		files = [args.input_path + i for i in files]
	else:
		f = open('./valid_scans_list.txt', 'r')
		files = [i.strip() for i in f.readlines()]
		f.close()

	output_dir = args.output_path
	shifts = list(range(0, 25, args.interval))

	shift_delta = lambda shift: np.array([[shift, 0, 0], [-shift, 0, 0], [0, shift, 0], [0, -shift, 0], [0, 0, shift], [0, 0, -shift]])
	shift_deltas = np.array(np.concatenate([shift_delta(i) for i in shifts]))
	shift_deltas = np.reshape(np.append([[0,0,0]], shift_deltas), [-1, 3])

	for curr, file in enumerate(files):
		h5_file = h5py.File(file, 'r')
		# The h5 file has to have three properties below
		raw = np.array(h5_file['raw'])
		label = np.array(h5_file['label'])
		w, h, d = np.array(h5_file['coor'])

		# Add Gaussian noise
		# noise = np.random.normal(0, 50, raw.shape)
		# raw += noise

		# Add Gaussian filter
		# label = gaussian_filter(label, sigma=0.5)
		
		for i in range(len(shift_deltas)):
			w1, h1, d1 = [w, h, d] + shift_deltas[i]
			raw_cropped = generate_slice(raw, w1, h1, d1)
			label_cropped = generate_slice(label, w1, h1, d1)
			if len(np.where(label == 1)[0]) != len(np.where(label_cropped == 1)[0]):
				continue
			new_h5_file = h5py.File(output_dir + file.split("/")[-1].split('.')[0] + '_' + str(i) + '.h5', 'w')
			new_h5_file.create_dataset('raw', data=raw_cropped)
			new_h5_file.create_dataset('label', data=label_cropped)
			new_h5_file.close()
		h5_file.close()
		print('%d/%d' % (curr, len(files)))

if __name__ == '__main__':
    main()