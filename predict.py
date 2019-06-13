import argparse
import os

import h5py
import numpy as np
import torch
import nibabel as nib
import cv2

from torch import nn as nn

from pcs_length import get_total_dist

from datasets.hdf5 import HDF5Dataset
from unet3d import utils
from unet3d.model import UNet3D
from unet3d.coor_model import CoorNet
from unet3d.losses import DiceCoefficient

from datasets.crop_window import generate_slice, recover_patch_4d, recover_patch, W, H, D
from datasets.align_acpc import preprocess_nifti, AFF

logger = utils.get_logger('UNet3DPredictor')


def predict(model, dataset, out_channels, device, x=None, y=None, z=None):
    """
    Return prediction masks by applying the model on the given dataset

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        dataset (torch.utils.data.Dataset): input dataset
        out_channels (int): number of channels in the network output
        device (torch.Device): device to run the prediction on

    Returns:
         probability_maps (numpy array): prediction masks for given dataset
    """
    # dimensionality of the the output (CxDxHxW)
    dataset_shape = dataset.raw.shape
    if len(dataset_shape) == 3:
        volume_shape = dataset_shape
    else:
        volume_shape = dataset_shape[1:]
    if isinstance(model, UNet3D):
        volume_shape = (W, H, D)
    probability_maps_shape = (out_channels,) + volume_shape
    # initialize the output prediction array
    probability_maps = np.zeros(probability_maps_shape, dtype='float32')

    # initialize normalization mask in order to average out probabilities
    # of overlapping patches
    normalization_mask = np.zeros(probability_maps_shape, dtype='float32')

    # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
    model.eval()
    with torch.no_grad():
        for patch, index in dataset:

            # If UNet, crop first
            if isinstance(model, UNet3D):
                patch = patch.cpu().numpy()
                patch = generate_slice(patch, x, y, z)
                patch = np.reshape(patch, (1,) + patch.shape)
                patch = torch.from_numpy(patch).to(device, dtype=torch.float32)

            # save patch index: (C,) + (D,H,W)
            channel_slice = slice(0, out_channels)
            index = (channel_slice,) + index

            # convert patch to torch tensor NxCxDxHxW and send to device
            # we're using batch size of 1
            patch = patch.view((1,) + patch.shape).to(device)

            # forward pass
            probs = model(patch)
            # convert back to numpy array
            probs = probs.squeeze().cpu().numpy()

            # If UNet, output normalised probability
            if isinstance(model, UNet3D):
                # for out_channel == 1 we need to expand back to 4D
                if probs.ndim == 3:
                    probs = np.expand_dims(probs, axis=0)
                # unpad in order to avoid block artifacts in the output probability maps
                # probs, index = utils.unpad(probs, index, volume_shape)
                # accumulate probabilities into the output prediction array
                probability_maps[index] += probs
                # count voxel visits for normalization
                normalization_mask[index] += 1
    

    if isinstance(model, UNet3D):
        return probability_maps / normalization_mask
    else:
        return probs

def pad(data):
    padded = np.zeros([192, 224, 192])
    padded[:182,:218,:182] = data
    return padded

def save_predictions(probability_maps, output_file, average_channels):
    """
    Saving probability maps to a given output H5 file. If 'average_channels'
    is set to True average the probability_maps across the the channel axis
    (useful in case where each channel predicts semantically the same thing).

    Args:
        probability_maps (numpy.ndarray): numpy array containing probability
            maps for each class in separate channels
        output_file (string): path to the output H5 file
        average_channels (bool): if True average out the channels in the probability_maps otherwise
            keep the channels separate
    """
    # logger.info(f'Saving predictions to: {output_file}')
    logger.info('Saving predictions to: ' + output_file)

    with h5py.File(output_file, "w") as output_h5:
        if average_channels:
            probability_maps = np.mean(probability_maps, axis=0)
        dataset_name = 'probability_maps'
        # logger.info(f"Creating dataset '{dataset_name}'")
        logger.info("Creating dataset " + dataset_name)
        output_h5.create_dataset(dataset_name, data=probability_maps, dtype=probability_maps.dtype, compression="gzip")


def main():
    parser = argparse.ArgumentParser(description='3D U-Net predictions')
    parser.add_argument('--cdmodel-path', required=True, type=str,
                        help='path to the coordinate detector model.')
    parser.add_argument('--model-path', required=True, type=str,
                        help='path to the segmentation model')
    parser.add_argument('--in-channels', type=int, default=1,
                        help='number of input channels (default: 1)')
    parser.add_argument('--out-channels', type=int, default=2,
                        help='number of output channels (default: 2)')
    parser.add_argument('--init-channel-number', type=int, default=64,
                        help='Initial number of feature maps in the encoder path which gets doubled on every stage (default: 64)')
    parser.add_argument('--layer-order', type=str,
                        help="Conv layer ordering, e.g. 'crg' -> Conv3D+ReLU+GroupNorm",
                        default='crg')
    parser.add_argument('--final-sigmoid',
                        action='store_true',
                        help='if True apply element-wise nn.Sigmoid after the last layer otherwise apply nn.Softmax')
    parser.add_argument('--test-path', type=str, nargs='+', required=True, help='path to the test dataset')
    parser.add_argument('--raw-internal-path', type=str, default='raw')
    parser.add_argument('--patch', type=int, nargs='+', default=None,
                        help='Patch shape for used for prediction on the test set')
    parser.add_argument('--stride', type=int, nargs='+', default=None,
                        help='Patch stride for used for prediction on the test set')
    parser.add_argument('--report-metrics', action='store_true',
                        help='Whether to print metrics for each prediction')
    parser.add_argument('--output-path', type=str, default='./output/',
                        help='The output path to generate the nifti file')
    

    args = parser.parse_args()

    # make sure those values correspond to the ones used during training
    in_channels = args.in_channels
    out_channels = args.out_channels
    # use F.interpolate for upsampling
    interpolate = True
    layer_order = args.layer_order
    final_sigmoid = args.final_sigmoid

    # Define model
    UNet_model = UNet3D(in_channels, out_channels,
                       init_channel_number=args.init_channel_number,
                       final_sigmoid=final_sigmoid,
                       interpolate=interpolate,
                       conv_layer_order=layer_order)
    Coor_model = CoorNet(in_channels)

    # Define metrics
    loss = nn.MSELoss(reduction='sum')
    acc = DiceCoefficient()
    
    logger.info('Loading trained coordinate detector model from ' + args.cdmodel_path)
    utils.load_checkpoint(args.cdmodel_path, Coor_model)

    logger.info('Loading trained segmentation model from ' + args.model_path)
    utils.load_checkpoint(args.model_path, UNet_model)

    # Load the model to the device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        logger.warning('No CUDA device available. Using CPU for predictions')
        device = torch.device('cpu')
    UNet_model = UNet_model.to(device)
    Coor_model = Coor_model.to(device)

    # Apply patch training if assigned
    if args.patch and args.stride:
        patch = tuple(args.patch)
        stride = tuple(args.stride)

    # Initialise counters
    total_dice = 0
    total_loss = 0
    count = 0
    tmp_created = False

    for test_path in args.test_path:
        if test_path.endswith('.nii.gz'):
            if args.report_metrics:
                raise ValueError("Cannot report metrics on original files.")
            # Temporary save as h5 file
            # Preprocess if dim != 192 x 224 x 192
            data = preprocess_nifti(test_path, args.output_path)
            logger.info('Preprocessing complete.')
            hf = h5py.File(test_path + '.h5', 'w')
            hf.create_dataset('raw', data=data)
            hf.close()
            test_path += '.h5'
            tmp_created = True
        if not args.patch and not args.stride:
            curr_shape = np.array(h5py.File(test_path, 'r')[args.raw_internal_path]).shape
            patch = curr_shape
            stride = curr_shape

        # Initialise dataset
        dataset = HDF5Dataset(test_path, patch, stride, phase='test', raw_internal_path=args.raw_internal_path)        

        file_name = test_path.split('/')[-1].split('.')[0]
        # Predict the centre coordinates
        x, y, z = predict(Coor_model, dataset, out_channels, device)

        # Perform segmentation
        probability_maps = predict(UNet_model, dataset, out_channels, device, x, y, z)
        res = np.argmax(probability_maps, axis=0)

        # Put the image batch back to mask with the original size
        res = recover_patch(res, x, y, z, dataset.raw.shape)

        # Extract LH and RH segmentations and write as file
        LH = np.zeros(res.shape)
        LH[int(res.shape[0]/2):,:,:] = res[int(res.shape[0]/2):,:,:]
        RH = np.zeros(res.shape)
        RH[:int(res.shape[0]/2),:,:] = res[:int(res.shape[0]/2),:,:]
        
        LH_img = nib.Nifti1Image(LH, AFF)
        RH_img = nib.Nifti1Image(RH, AFF)
        nib.save(LH_img, args.output_path + file_name + '_LH.nii.gz')
        nib.save(RH_img, args.output_path + file_name + '_RH.nii.gz')
        logger.info('File saved to ' + args.output_path + file_name + '_LH.nii.gz')
        logger.info('File saved to ' + args.output_path + file_name + '_RH.nii.gz')
        
        if tmp_created:
            os.remove(test_path)

        if args.report_metrics:
            count += 1

            # Compute coordinate accuracy
            # Coordinate evaluation disabled by default, since not all data have coordinate information
            # coor_dataset = HDF5Dataset(test_path, patch, stride, phase='val', raw_internal_path=args.raw_internal_path, label_internal_path='coor')
            # coor_target = coor_dataset[0][1].to(device)
            # coor_pred_tensor = torch.from_numpy(np.array([x, y, z])).to(device)
            # curr_coor_loss = loss(coor_pred_tensor, coor_target)
            # total_loss += curr_coor_loss
            # logger.info('Current coordinate loss: %f' % (curr_coor_loss))

            # Compute segmentation Dice score
            label_dataset = HDF5Dataset(test_path, patch, stride, phase='val', raw_internal_path=args.raw_internal_path, label_internal_path='label')
            label_target = label_dataset[0][1].to(device)
            res_dice = probability_maps
            new_shape = np.append(res_dice.shape[0], np.array(label_target.size()))
            res_dice = recover_patch_4d(res_dice, x, y, z, new_shape)
            pred_tensor = torch.from_numpy(res_dice).to(device).float()
            label_target = label_target.view((1,) + label_target.shape)
            curr_dice_score = acc(pred_tensor, label_target.long())
            total_dice += curr_dice_score
            logger.info('Current Dice score: %f' % (curr_dice_score))

            # Compute length estimation
            logger.info('RH length: ' + str(get_total_dist(res[:int(res.shape[0]/2),:,:])))
            logger.info('LH length: ' + str(get_total_dist(res[int(res.shape[0]/2):,:,:])))
    
    if args.report_metrics:       
        # logger.info('Average loss: %f.' % (total_loss/count))
        logger.info('Average Dice score: %f.' % (total_dice/count))

if __name__ == '__main__':
    main()
