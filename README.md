# SP-WNet

Unsupervised Image Segmentation with Dense Representative Learning and Sparse Labelling


## Synopsis (paper abstract)
Fully unsupervised semantic segmentation of images has been a challenging problem 
in computer vision due to its non-convexity and data insufficiency (e.g., only
one target image available). Many deep learning models have been developed
for this task, from dense or pixel-based to sparse or graph-based ones, mostly
using representative learning guided by certain loss functions towards segmentation. 
In this paper, we conduct dense representative learning using an existing
fully-convolutional autoencoder; based on a predetermined oversegmentation, the
learned dense features are reduced to a feature graph where segmentation can be
encouraged from three aspects: normalised cut, similarity and continuity. Our
model can be trained with one or several images. To alleviate overfitting, we
compute the reconstruction loss using size-varying random patches taken from
the input image(s). We show that the model trained with one or a few images
can be robust for predicting other images with similar semantic contents, meaning
that the model trained in 2D can be used to segment a 3D image or a video. Our
experiments include general images and videos as well as 3D tomographic images
from neutron and X-ray scattering.

The full paper will be released soon. Currently cite the code as follow:

```
@misc{spwnet2022,
    title  = {SP-WNet: Unsupervised Image Segmentation with Dense Representative Learning and Sparse Labelling},
    author = {Kuangdai Leng, Robert Atwood, Winfried Kockelmann, Deniza Chekrygina, Jeyan Thiyagalingam},
    url    = {https://github.com/stfc-sciml/sp-wnet-seg},
    year   = {2022}
}
```





## Installation


```sh
pip install -r requirements.txt
```

## User guide

Unsupervised segmentation of a video can be performed with the following steps,
an example taken from [demo-cheetah.ipynb](demo-cheetah.ipynb). 
Just skip the first and the last steps for a single-image task.

#### Step 1: Select input image(s)
Suppose we have following target video for segmentation:

<img src="https://github.com/stfc-sciml/sp-wnet-seg/blob/main/readme-resources/video.gif" width="70%">

The fist step is to select a few representative clips, such as the following four. Here we
rescale the image size by 1/3 and applied some unsharp masking.

<img src="https://github.com/stfc-sciml/sp-wnet-seg/blob/main/readme-resources/input.png" width="70%">


#### Step 2: Sample patches from each input image
We train the autoencoder (a WNet) with smaller patches that are sampled from the original images and augmented 
(flipped or rotated).
This can be done conveniently using the function `sample_patch_datasets()` from `src/utils.py`.
For example, the following figure show the patches sampled from one of the above input images.
They are in four different sizes, each size with 8 mini-batches: 
* 64x64, with 9 patches per mini-batch
* 64x96, with 6 patches per mini-batch
* 96x64, with 6 patches per mini-batch
* 96x96, with 4 patches per mini-batch

<img src="https://github.com/stfc-sciml/sp-wnet-seg/blob/main/readme-resources/patches.png" width="90%">

 
#### Step 3: Generate superpixels for each input image
Segmentation will be performed based on predetermined superpixels, which can be done using an oversegmentation
algorithm such as SLIC. Our deep model supports a large number of superpixels, so this step can be done
trivially. Surely, a more careful oversegmentation that yields fewer superpixels can facilitate
the training of the deep model.  The following figure shows the oversegmentation for 
one of the above input images, containing more than 13000 superpixels.

<img src="https://github.com/stfc-sciml/sp-wnet-seg/blob/main/readme-resources/slic.png" width="70%">

**Note**: our deep model also supports pixel-wise or dense segmentation, for which this step can be skipped.

#### Step 4: Train a `WNet`
The data are ready. Now one can create a `WNet`, as defined in `src/wnet.py`, and use the function 
`train_sp_wnet()` from `src/train_predict.py` to train the `WNet` model.
Several network and loss hyperparameters can be specified. 
In general, 
hyperparameter tuning tends to be more difficult and laborious for unsupervised learning.
Our code can visualize the segmentation results during training, which may help accelerating
hyperparameter tuning.


#### Step 5: Segment all the clips with the trained `WNet`
Once training is done, the trained `WNet` can (hopefully) be used to segment all the clips 
from the video with a good accuracy. The results for this example are shown below.

<img src="https://github.com/stfc-sciml/sp-wnet-seg/blob/main/readme-resources/mark.gif" width="70%">



## Funding and Support

This work was supported by the ALC Project *AIDA-NXtomo (Data fusion of neutron and X-ray tomographic data)*.

