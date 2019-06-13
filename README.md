
# Volumetric Segmentation and Characterisation of the Paracingulate Sulcus on MRI Scans

The implementation of my dissertation, the work is an extension of [pytorch-3dunet](https://github.com/wolny/pytorch-3dunet).

## Getting Started

### Dependencies
- pytorch (0.4.1+)
- torchvision (0.2.1+)
- tensorboardx (1.6+)
- h5py

Setup a new conda environment with the required dependencies via:
```
conda create -n 3dunet pytorch torchvision tensorboardx h5py -c conda-forge -c pytorch
``` 
Activate newly created conda environment via:
```
source activate 3dunet
```
## TL;DR
Download the pre-trained coordinate detector and the segmentation network model:

[Coordinate Detector](https://drive.google.com/file/d/1wV7utlnMamF_phVqkk4CAkidnPVDx7tQ/view?usp=sharing)

[Segmentation Network](https://drive.google.com/file/d/1S4CvOQj3KNVaVfUroVk7M66jKbiEmX_q/view?usp=sharing)

To predict on the NIfTI file, run `predict.py` with the following command:
```
python predict.py --model-path ./seg.pytorch --cdmodel-path ./cd.pytorch --test-path ./file.nii.gz
```
The command will create two masks for LH and RH in the path `./output/`.

## Data Preparation
### Data Preprocessing
With a given scan, if the size does not match with the intended size, preprocessing can be performed. The script is available in `./datasets/align_acpc.py`. The scan is resampled to 192 x 224 x 192 and 1 x 1 x 1 mm voxel first, and is then aligned to the ACPC plane using a 6 DOF alignment via FSL commands based on the template file `./datasets/template.nii.gz`. The preprocessing step will be automatically performed during prediction, where the output segmentation mask will be based on the processed scan. However, for preparing the training data, the implemented function needs to be manually added to the conversion script mentioned in the upcoming section.

### Data Conversion
NIfTI files need to be converted to HDF5 format first **for training**. An example can be found in `./datasets/h5_converter.py`, which can be executed to generate h5 file for all provided MRI scans. Similar conversion can be done by modifying the file to enable conversion on other files.

Each generated h5 file has three datasets, `raw`, `label`, and `coor`. The `raw` and `label` sets correspond to the raw image and the label of the scan, and zero-padding is performed to resize the image to 192 x 224 x 192 (divisble by 32). Lastly, the `coor` dataset contains the centre coordinate `(x, y, z)` for each scan. As not all scans have the PCS, a threshold is applied to only generate the `coor` dataset for scans have more PCS voxels than the threshold.

### Data Augmentation
After conversion, the augmentation can be performed:
```
python ./datasets/augmentation.py --input-path ./PCS_h5_files/* --output-path ./aug_PCS_files/ --interval 4
```
The `input-path` and `output-path` arguments indicate the path for input and output, and `interval` is the step size of shift offset used when performing, which can indirectly affect the augmented sample size.

For example, when an interval of 4 is applied, shifts will be generated from range(0, 25, 4), and augmentation will be made on moving the centre coordinate towards different directions by 0, 4, 8, ... voxels, and crop the surrounding image based on the sliding window size.

#### IMPORTANT
The file to be augmented must contain the `coor` dataset.

## Supported Losses

### Loss functions
- **wce** - _WeightedCrossEntropyLoss_ (see 'Weighted cross-entropy (WCE)' in the above paper for a detailed explanation)
- **ce** - _CrossEntropyLoss_ (one can specify class weights via `--loss-weight <w_1 ... w_k>`)
- **pce** - _PixelWiseCrossEntropyLoss_ (once can specify not only class weights but also per pixel weights in order to give more/less gradient in some regions of the ground truth)
- **bce** - _BCELoss_ (one can specify class weights via `--loss-weight <w_1 ... w_k>`)
- **dice** - _DiceLoss_ standard Dice loss (see 'Dice Loss' in the above paper for a detailed explanation). Note: if your labels in the training dataset are not very imbalance
e.g. one class having at lease 3 orders of magnitude more voxels than the other use this instead of `GDL` since it worked better in my experiments.
- **gdl** - _GeneralizedDiceLoss_ (one can specify class weights via `--loss-weight <w_1 ... w_k>`)(see 'Generalized Dice Loss (GDL)' in the above paper for a detailed explanation)

## Train
```
usage: train.py [-h] [--checkpoint-dir CHECKPOINT_DIR] [--in-channels
                IN_CHANNELS] [--out-channels OUT_CHANNELS]
                [--init-channel-number INIT_CHANNEL_NUMBER]
                [--layer-order LAYER_ORDER] [--loss LOSS]
                [--loss-weight LOSS_WEIGHT [LOSS_WEIGHT ...]]
                [--ignore-index IGNORE_INDEX] [--curriculum] [--final-sigmoid]
                [--epochs EPOCHS] [--iters ITERS] [--patience PATIENCE]
                [--learning-rate LEARNING_RATE] [--weight-decay WEIGHT_DECAY]
                [--validate-after-iters VALIDATE_AFTER_ITERS]
                [--log-after-iters LOG_AFTER_ITERS] [--resume RESUME]
                --train-path TRAIN_PATH [TRAIN_PATH ...] 
                --train-patch TRAIN_PATCH [TRAIN_PATCH ...]
                --train-stride TRAIN_STRIDE [TRAIN_STRIDE ...]
                [--raw-internal-path RAW_INTERNAL_PATH]
                [--label-internal-path LABEL_INTERNAL_PATH]
                [--transformer TRANSFORMER]
                [--network NETWORK]

UNet3D training

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint-dir CHECKPOINT_DIR
                        checkpoint directory
  --in-channels IN_CHANNELS
                        number of input channels (default: 1)
  --out-channels OUT_CHANNELS
                        number of output channels (default: 2)
  --init-channel-number INIT_CHANNEL_NUMBER
                        Initial number of feature maps in the encoder path
                        which gets doubled on every stage (default: 64)
  --layer-order LAYER_ORDER
                        Conv layer ordering, e.g. 'crg' ->
                        Conv3D+ReLU+GroupNorm
  --loss LOSS           Which loss function to use for segmentation network. 
            Possible values: [bce, ce, wce, dice]. Where bce -
                        BinaryCrossEntropyLoss (binary classification only),
                        ce - CrossEntropyLoss (multi-class classification),
                        wce - WeightedCrossEntropyLoss (multi-class
                        classification), dice - GeneralizedDiceLoss
                        (multi-class classification)
  --loss-weight LOSS_WEIGHT [LOSS_WEIGHT ...]
                        A manual rescaling weight given to each class. Can be
                        used with CrossEntropy or BCELoss. E.g. --loss-weight
                        0.3 0.3 0.4
  --ignore-index IGNORE_INDEX
                        Specifies a target value that is ignored and does not
                        contribute to the input gradient
  --curriculum          use simple Curriculum Learning scheme if ignore_index
                        is present
  --final-sigmoid       if True apply element-wise nn.Sigmoid after the last
                        layer otherwise apply nn.Softmax
  --epochs EPOCHS       max number of epochs (default: 500)
  --iters ITERS         max number of iterations (default: 1e5)
  --patience PATIENCE   number of epochs with no loss improvement after which
                        the training will be stopped (default: 20)
  --learning-rate LEARNING_RATE
                        initial learning rate (default: 0.0002)
  --weight-decay WEIGHT_DECAY
                        weight decay (default: 0.0001)
  --validate-after-iters VALIDATE_AFTER_ITERS
                        how many iterations between validations (default: 100)
  --log-after-iters LOG_AFTER_ITERS
                        how many iterations between tensorboard logging
                        (default: 100)
  --resume RESUME       path to latest checkpoint (default: none); if provided
                        the training will be resumed from that checkpoint
  --train-path TRAIN_PATH [TRAIN_PATH ...]
                        paths to the training datasets, e.g. --train-path <path1> <path2>
  --train-patch TRAIN_PATCH [TRAIN_PATCH ...]
                        Patch shape for used for training
  --train-stride TRAIN_STRIDE [TRAIN_STRIDE ...]
                        Patch stride for used for training
  --raw-internal-path RAW_INTERNAL_PATH
  --label-internal-path LABEL_INTERNAL_PATH
  --transformer TRANSFORMER
                        data augmentation class
  --network NETWORK
            which network to train, cd for coordinate detector
            and seg for segmentation network.
```

For direct training on the whole image without using patch based training, simply assign the `train-patch` and `train-stride` arguments as the dimension of the image.

Train on coordinate detector:
```
python train.py --checkpoint-dir ./ckpt/ --epoch 50 --learning-rate 0.0002 --train-path ./PCS_data_h5/* --train-patch 192 224 192 --train-stride 192 224 192 --label-internal-path coor --network cd
```

Train on segmentation network using Dice loss:
```
python train.py --checkpoint-dir ./ckpt/ --epoch 10 --learning-rate 0.0002 --train-path ./PCS_data_h5/* --train-patch 63 77 93 --train-stride 63 77 93 --network seg --loss dice
```
 
To resume training the segmentation from the last checkpoint:
```
python train.py --resume ./ckpt/seg_ckpt.pytorch --epoch 10 --learning-rate 0.0002 --train-path ./PCS_data_h5/* --train-patch 63 77 93 --train-stride 63 77 93 --network seg --loss dice
```

### IMPORTANT
In order to train with `BinaryCrossEntropy` the label data has to be 4D! (one target binary mask per channel). `--final-sigmoid` has to be given when training the network with `BinaryCrossEntropy`
(and similarly `--final-sigmoid` has to be passed to the `predict.py` if the network was trained with `--final-sigmoid`)

`DiceLoss` and `GeneralizedDiceLoss` support both 3D and 4D target (if the target is 3D it will be automatically expanded to 4D, i.e. each class in separate channel, before applying the loss).



## Test
```
usage: predict.py [-h] --cdmodel-path MODEL_PATH --model-path MODEL_PATH 
                  [--in-channels IN_CHANNELS] [--out-channels OUT_CHANNELS]
                  [--init-channel-number INIT_CHANNEL_NUMBER]
                  [--layer-order LAYER_ORDER] [--final-sigmoid] --test-path
                  TEST_PATH [--raw-internal-path RAW_INTERNAL_PATH] --patch
                  PATCH [PATCH ...] --stride STRIDE [STRIDE ...]
                  [--report-metrics] [--output-path OUTPUT_PATH]

3D U-Net predictions

optional arguments:
  -h, --help            show this help message and exit
  --cdmodel-path MODEL_PATH
                        path to the coordinate detector model
  --model-path MODEL_PATH
                        path to the segmentation model
  --in-channels IN_CHANNELS
                        number of input channels (default: 1)
  --out-channels OUT_CHANNELS
                        number of output channels (default: 2)
  --init-channel-number INIT_CHANNEL_NUMBER
                        Initial number of feature maps in the encoder path
                        which gets doubled on every stage (default: 64)
  --layer-order LAYER_ORDER
                        Conv layer ordering, e.g. 'crg' ->
                        Conv3D+ReLU+GroupNorm
  --final-sigmoid       if True apply element-wise nn.Sigmoid after the last
                        layer otherwise apply nn.Softmax
  --test-path TEST_PATH
                        path to the test dataset
  --raw-internal-path RAW_INTERNAL_PATH
  --patch PATCH [PATCH ...]
                        Patch shape for used for prediction on the test set
  --stride STRIDE [STRIDE ...]
                        Patch stride for used for prediction on the test set
  --report-metrics
              Whether to print metrics for each prediction
  --output-path OUTPUT_PATH
              The output path to generate the nifti file
```

To predict and test on h5 files, the following command can be executed to report metrics (e.g. Dice score, loss...) and save predicted segmentation as two NIfTI files for each scan:
```
python predict.py --model-path ./ckpt/seg_ckpt.pytorch --cdmodel-path ./ckpt/cd_ckpt.pytorch --test-path ./PCS_data_h5/* --report-metrics
```

To simply generate NIfTI file of prediction:
```
python predict.py --model-path ./seg.pytorch --cdmodel-path ./cd.pytorch --test-path ./file.nii.gz
```
### IMPORTANT
Image preprocessing is performed when the given NIfTI file has a mismatched dimension, along with the predicted segmentation mask for LH and RH, upon preprocessing the processed NIfTI scan file will also be generated in the output path.