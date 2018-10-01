# ShapeNet-Intrinsics

This is implementation for [Learning Non-Lambertian Object Intrinsics across ShapeNet Categories](https://arxiv.org/abs/1612.08510)

You might be interested in the synthetic dataset we used in the paper. The entire dataset takes more than 1T for HDR images, and 240G for even compressed .jpg images. So it is hard to share it online, and we are still working on it;)

However, you can still check the [rendering scripts](render), which can generate the dataset and do even more things for your own need, e.g. depth and normal images. [Training and testing scripts](train) are implemented in [torch](http://torch.ch/).

#### Downloads

Trained torch model and HDR environment map can be accessed [here](https://1drv.ms/f/s!ApfQp_rip6el-X-neX32NGAE_aiC).

Note: there are two models(model.t7 and model_old.t7). Torch updated its API on SpatialUpSamplingBilinear, which is used in the model. The old one is trained with old API(maybe before Oct 2016), and model.t7 works on current version of torch.

model.t7 is trained for 1M steps, which is observed overfitting on syn data.
model_old.t7 is trained for 450k steps. It performs worse than model.t7 on synthetic data, but for some real data, it might produce better result. To try old version of model, simply uncomment line 10 in Test.lua (-- require 'Patch').

#### Torch is outdated...

Recently I found that people switch to tensorflow and pytorch, while torch is not active. I did spend some time to try to migrate the work to tensorflow, but I found the exact same network structure in tensorflow is not working. The major problem is that I used ReLU in the network. In tensorflow, the network produces all black image, and pixel values and gradients are clamped. I tried different LR and optimizer, but cannot make it work. In torch, everything is OK. Although I can use LeakyReLU in tensorflow, it is different from the original version. I doubt there is some difference in the internal implementation of these frameworks. I don't have enough time to make it works, but I maganed to compile torch and run the model under a relative new platform: **ubuntu 18.04, cuda 9.0 and cudnn 7**, which I think is acceptable for most people.

#### Running the code...

Here is the note for run the code with torch under ubuntu 18.04, cuda 9.0 and cudnn 7.

1. Clone the torch repo as usual
```
git clone https://github.com/torch/distro.git ~/torch --recursive
```
2. Modify the install-deps, in line 178, 'sudo apt-get install -y python-software-properties'. The package in ubuntu 18.04 is replaced by 'software-properties-common', you can replace it or manually install the package.
3. Set gcc compiler to gcc-6, which is the maximun version for cuda-9.0. First install gcc-6 with apt, and then change default gcc to gcc-6.
```
sudo apt install gcc-6
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 10
sudo update-alternatives --config gcc
```
1. Install.
```
export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"
./install.sh
```
2. If everything is OK, torch would run...on cudnn5...Now we need to switch to cudnn7.
```
cd extra/cudnn
git fetch
git checkout R7
luarocks make cudnn-scm-1.rockspec 
```
Now you can probably run torch.