# ShapeNet-Intrinsics

This is implementation for [Learning Non-Lambertian Object Intrinsics across ShapeNet Categories](https://arxiv.org/abs/1612.08510)

You might be interested in the synthetic dataset we used in the paper. The entire dataset takes more than 1T for HDR images, and 240G for even compressed .jpg images. So it is hard to share it online, and we are still working on it;)

However, you can still check the [rendering scripts](render), which can generate the dataset and do even more things for your own need, e.g. depth and normal images. [Training and testing scripts](train) are implemented in [torch](http://torch.ch/).
