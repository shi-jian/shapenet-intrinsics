# Training/Testing scripts

This directory provides network structure, criterion, training and testing scripts.

## Train

Some parameters:
* **-data_root** specify the root directory of ShapeNet rendering images.
* **-model_list** specify the .csv file containing model id and dataset separation downloaded from [ShapeNet](http://shapenet.cs.stanford.edu/shapenet/obj-zip/SHREC16/all.csv) website.
* **-env_list** specify the environment map list file.
* **-outdir** specify the output directory for saving snapshots. Default is current directory.

## Test

Testing scripts is quite simple. It accepts 5 parameters.

* **-input** specify the input image file.
* **-mask** specify the mask file.
* **-model** specify the trained model file.
* **-outdir** specify the output directory, default is current directory.
* **-gpu** 0 is for running on CPU. Default is using GPU.


The script would output 5 images including albedo.png, shading.png, specular.png, as well as input.png and mask.png under outdir.
