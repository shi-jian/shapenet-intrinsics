# ShapeNet-Intrinsics

This is implementation for [Learning Non-Lambertian Object Intrinsics across ShapeNet Categories](https://arxiv.org/abs/1612.08510)

You might be interested in the synthetic dataset we used in the paper. The entire dataset takes more than 1T for HDR images, and 240G for even compressed .jpg images. So it is hard to share it online, and we are still working on it;)

However, you can still check the [rendering scripts](render), which can generate the dataset and do even more things for your own need, e.g. depth and normal images. [Training and testing scripts](train) are implemented in [torch](http://torch.ch/).

#### Downloads
* [trained model](http://share.shijian.org.cn/shapenet/intrinsics/model.t7) you can try our trained model (450k iterations).
* [environment maps](http://share.shijian.org.cn/shapenet/intrinsics/envmap.zip) although you can find all envmaps [here](http://www.hdrlabs.com/sibl/archive.html), we prepared an archive file for you. Please note that the envmaps in the archive are down-sized.
* [mitsuba plugin](http://share.shijian.org.cn/shapenet/render/shapenet.dll) here is a compiled Windows DLL mitsuba plugin for loading and rendering ShapeNet .obj models.
