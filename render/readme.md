# Rendering Script

In this directory we provide rendering configuration template for [mitsuba-shapenet](https://github.com/shi-jian/mitsuba-shapenet]) to render different components of a ShapeNet model.

### Configuration templates

* [template-i.xml](template-i.xml) configuration for rendering image.
* [template-a.xml](template-a.xml) for albedo.
* [template-s.xml](template-s.xml) for shading.
* [template-r.xml](template-r.xml) for specular. We use homogeneous specular for the entire model.
* [template-d.xml](template-d.xml) for depth, which is used to generate mask.

Note: In the experiments, we rendered albedo/shading/specular and then synthesized image by I=A*S+R. Depth is used to generate object mask.

It might be useful to look into albedo/depth configuration file if you want to render other 'field' in mitsuba, such as normal.

### Script for generating rendering scripts

[gen_script.py](gen_script.py) is used to generate rendering and synthesize scripts. Please set following environment:

* MITSUBA
points to mitsuba renderer executable (e.g. mitsuba.exe in windows).

* SHAPENET_ROOT
 the directory contains extracted ShapeNet models.

* ENVMAP_ROOT
the directory contains environment maps, with a 'list.txt' file. Each line of the list file contains an environment map filename.

* RENDER_ROOT
the directory to put rendering scripts and results.


Recently ShapeNet released an official dataset separation. The script would automatically download the model [list](http://shapenet.cs.stanford.edu/shapenet/obj-zip/SHREC16/all.csv) from ShapeNet, which contains models, categories, uuid and data separation. Then it would generate output directories for models under RENDER_ROOT, as well as two scripts: render.bat/render.sh and synthesize.bat/synthesize.sh.

* render.bat: render albedo/shading/specular/depth in HDR images.
* synthesize.bat: generate mask image from depth, convert HDR to LDR for albedo/shading/specular(for saving disk space), generate image by I=A*S+R. [ImageMagick](http://www.imagemagick.org) is required for image synthesizing.

Then, you can run these scripts under their directory. We strongly recommend to render on a cluster. Rendering for a single model under 92 environment maps takes about 45 min on an i7-2600 old PC.
