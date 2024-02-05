# Centroid-based method

## Introduction
This code is based on the implementation of [ReID Strong Baseline](https://github.com/michuanhaohao/reid-strong-baseline). You can refer to it for more details.

[[Journal Version(TMM)]](https://ieeexplore.ieee.org/document/8930088)
[[PDF]](http://openaccess.thecvf.com/content_CVPRW_2019/papers/TRMTMCT/Luo_Bag_of_Tricks_and_a_Strong_Baseline_for_Deep_Person_CVPRW_2019_paper.pdf)


The main contributions introduced are:
* Pre-training the backbone on the [LUPerson](https://github.com/DengpanFu/LUPerson) dataset.
* Integration of the novel data augmentation technique called [Random Color Dropout](https://github.com/finger-monkey/Data-Augmentation).
* Integration of a new triplet loss based on centroids.
* Addition of a new centroid-based metric to be used during the retrieval stage.
* Addition of a more valuable experimental protocol based on cross-dataset evaluation, and including the CUHK03 dataset.
<br><br>
## Dependencies

* PyTorch >= 0.4
* PyTorch Ignite = 0.1.2
* Yacs
* OpenCV
* Numpy 
* PIL
<br><br>
## Dataset organization
The supported datasets are Market-1501, DukeMTMC-reID, CUHK03, and ALL. To add your custom dataset, it is important that it follows the following data organization:
* bounding_box_test
    * pid_ccamid_frame.jpg
    * ...
* bounding_box_train
    * pid_ccamid_frame.jpg
    * ...
* query
    * pid_ccamid_frame.jpg
    * ...

Note that images names can also follow other naming formats; the important thing is that the beginning of the file is of the tpye
'x_cy' where 'x' is the PID representing the identity of the depicted person, and 'y' is the CAMID representing the camera identifier used to capture the image.
<br><br>
## Installation
Download the attached code. Then, modify the variable indicating the dataset directory in `data/datasets/dataset_file.py` in your `directory_dir`. 
For example, if you want to use DukeMTMC-reID:
1. Open the `data/datasets/dukemtmcreid.py` file.
2. Find the statement `dataset_dir = 'DukeMTMC'`.
3. Replace the string with `dataset_dir = 'YOUR_DUKEMTMC_DIRECTORY_NAME`.
<br><br>
## Parameters
For the experiments, you can consider the following parameters, all of which are available and documented in the `config/defaults.py` file.
The complete set of parameters is summarized here:
* **MODEL**
    * **DEVICE**: can be 'cuda' or 'cpu'.
    * **DEVICE_ID**: selects the index of the device. 
    * **NAME**: name of the backbone to use, can be 'resnet50', 'resnet101' or 'efficientnet'.
    * **LAST_STRIDE**: last stride of the backbone to use, can be 1 or 2.
    * **PRETRAIN_PATH**: specifies the path to pre-trained weights of the model.
    * **PRETRAIN_CHOICE**: use ImageNet or LUPeron pre-trained model to initialize backbone. it can be 'luperson' or 'imagenet'.
    * **NECK**: if train with BNNeck, options: 'bnneck' or 'no'.
    * **IF_WITH_CENTER**: if train loss include center loss, options: 'yes' or 'no'.
    * **METRIC_LOSS_TYPE**: the loss type of metric loss. Use 'XE-tri-mg' to apply the new centroid-based loss.
    * **IF_LABELSMOOTH**: if train with label smooth, options: 'on' or 'off'. 
* **INPUT**
    * **SIZE_TRAIN**: image size during training.
    * **SIZE_TEST**: image size during test.
    * **FLIP_PROB**: random probability for image horizontal flip.
    * **RE_PROB**: random probabiliy for random erasing.
    * **PIXEL_MEAN**: mean to be used for image normalization.
    * **PIXEL_STD**: standard deviation to be used for image normalization.
    * **PADDING**: value of padding size.
    * **RCD_PROB**: random probability for random color dropout.
    * **CJ_PROB**: random probability for color jitter.
* **DATASETS**
    * **NAMES**: list of the dataset names for training.
    * **ROOT_DIR**: root directory where datasets should be used.
* **DATALOADER**
    * **NUM_WORKERS**: number of data loading threads.
    * **SAMPLER**: sampler for data loading, can be 'softmax' or 'triplet_softmax'.
    * **NUM_INSTANCE**: number of instances for one batch.
* **SOLVER**
    * **OPTIMIZER_NAME**: name of optimizer.
    * **MAX_EPOCHS**: number of max epoches.
    * **BASE_LR**: base learning rate.
    * **BIAS_LR_FACTOR**: factor of learning bias.
    * **MOMENTUM**: momentum.
    * **MARGIN**: triplet loss margin value.
    * **WARMUP_FACTOR**: LR warmup factor.
    * **WARMUP_METHOD**: method of LR warmup, option: 'constant' or 'linear'.
    * **CHECKPOINT_PERIOD**: epoch number of saving checkpoints.
    * **LOG_PERIOD**: iteration of display training log.
    * **EVAL_PERIOD**: epoch number of validation.
    * **IMS_PER_BATCH**: number of images per batch during training.
    * **...**
* **TEST**
    * **IMS_PER_BATCH**: number of images per batch during test.
    * **RE_RANKING**: if test with re-ranking, can be 'yes' or 'no'.
    * **WEIGHT**: path to trained model.
    * **NECK_FEAT**: which feature of BNNeck to be used for test, before or after BNNeck, options: 'before' or 'after'.
    * **FEAT_NORM**: whether feature is nomalized before test, if yes, it is equivalent to cosine distance.
    * **WITH_CENTROIDS**: if test with centroid-based metric, options: 'yes', 'no'.
* **OUTPUT_DIR**: path to checkpoint and saved log of trained model.

To run the already conducted experiments, you can check the ones available in the `configs/experimental_test` directory and its sub-directories. If you want to create your custom experiment, simply create a new `.yml` file in the `configs` directory and set the parameters as you prefer.

<br><br>
## Train
Let `configs/experimental_test/file.yml` be your configuration file. To run the model training, simply execute the followings.

Navigate to your directory containing the code, specifically the one that includes the 'tools' folder:
```
cd `your_code_directory`
```
Then, execute the train script:
```
python3 tools/train.py --config_file="configs/experimental_test/file.yml" 
```
If you want to set other parameters at this point, you can do so with:

```
python3 tools/train.py --config_file="configs/experimental_test/file.yml" \ 
--PARAMETER_1="parameter_1_value" --PARAMETER_2="parameter_2_value" 
```

<br><br>
## Test
Let `configs/experimental_test/test_set/file.yml` be your configuration file. To run the model test, simply execute the followings.

Navigate to your directory containing the code, specifically the one that includes the 'tools' folder:
```
cd `your_code_directory`
```
Then, execute the test script:
```
python3 tools/test.py --config_file="configs/experimental_test/test_set/file.yml" 
```
If you want to set other parameters at this point, you can do so with:

```
python3 tools/test.py --config_file="configs/experimental_test/test_set/file.yml" \ 
--PARAMETER_1="parameter_1_value" --PARAMETER_2="parameter_2_value" 
```
Remember to specify the path to the pretrained model weights `.pth` file as the value for the parameter TEST.WEIGHT. You can do this either from the command line (as indicated in the above instructions) or in the `.yml` configuration file.
To enable the centroid-based metric, set TEST.WITH_CENTROIDS to 'yes'.

<br><br>
## Quotes
### Bag of Tricks and a Strong Baseline for Deep Person Re-Identification

- **Title**: Bag of Tricks and a Strong Baseline for Deep Person Re-Identification
- **Author**: Luo, Hao and Gu, Youzhi and Liao, Xingyu and Lai, Shenqi and Jiang, Wei
- **Book title**: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops
- **Month**: June
- **Year**: 2019

### A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification

- **Title**: A Strong Baseline and Batch Normalization Neck for Deep Person Re-identification
- **Author**: H. {Luo} and W. {Jiang} and Y. {Gu} and F. {Liu} and X. {Liao} and S. {Lai} and J. {Gu}
- **Journal**: IEEE Transactions on Multimedia
- **Book title**: The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops
- **Year**: 2019
- **Pages**: {1-1}
- **doi**: 10.1109/TMM.2019.2958756
- **ISSN**: 1941-0077