# generate rendering scripts
import os
import platform
import random
import math
import urllib.request

## for testing...
#os.environ["MTISUBA"] = "D:/Develope/Project/mitsuba_plugins/build/mitsuba/binaries/MinSizeRel/mitsuba.exe"
#os.environ["SHAPENET_ROOT"] = "E:/ShapeNet/ShapeNetCore.v1/models"
#os.environ["RENDER_ROOT"] = "E:/ShapeNet/Render"
#os.environ["ENVMAP_ROOT"] = "E:/ShapeNet/Envmap"

# fix random seed
random.seed(0)
# generate a random viewpoing
def RandomView():
    theta = random.random() * math.pi * 0.5 # theta should be in [0, 0.5PI)
    phi = random.random() * math.pi * 2.0 # [0, 2PI)
    x = math.sin(theta) * math.cos(phi)
    y = math.cos(theta)
    z = math.sin(theta) * math.sin(phi)

    return "\"%f,%f,%f\"" % (x * 2, y * 2, z * 2)

# rendering options
# useful options: -q quite, -x skip exists
options = "-q"

# templates
script_dir = os.path.dirname(os.path.abspath(__file__))
template_i = os.path.join(script_dir, "template-i.xml")
template_a = os.path.join(script_dir, "template-a.xml")
template_s = os.path.join(script_dir, "template-s.xml")
template_r = os.path.join(script_dir, "template-r.xml")
template_d = os.path.join(script_dir, "template-d.xml")

# the mitsuba executable binary
MITSUBA = os.environ["MTISUBA"]
# the ShapeNet model repository directory
SHAPENET_ROOT = os.environ["SHAPENET_ROOT"]
# the environment map root folder
ENVMAP_ROOT = os.environ["ENVMAP_ROOT"]
# where to put rendering output
RENDER_ROOT = os.environ["RENDER_ROOT"]

# first we create a rendering output directory
if not os.path.exists(RENDER_ROOT):
    os.makedirs(RENDER_ROOT)

# download all.csv from shapenet, which contains a separation of dataset
list_url = "http://shapenet.cs.stanford.edu/shapenet/obj-zip/SHREC16/all.csv"
list_file = os.path.join(RENDER_ROOT, "dataset.csv")
if not os.path.exists(list_file):
    print("Download model list from ShapeNet website...")
    u = urllib.request.urlretrieve(list_url, list_file)

# load environment map list
# suppose there is a list.txt under ENVMAP_ROOT
envmaps = []
for line in open(os.path.join(ENVMAP_ROOT, "list.txt")):
    env_file = os.path.join(ENVMAP_ROOT, line[:-1])
    #print(env_file)
    if os.path.exists(env_file):
        envmaps.append(env_file)

model_list = open(list_file, 'r')

first_row = True
for row in model_list:
    # split the first row
    if first_row:
        first_row = False
        continue

    cols = row.split(",")

    idx = cols[0]       # model index
    category = cols[1]  # model category
    uuid = cols[3]      # model ID

    # the model .obj file
    model_file = os.path.join(SHAPENET_ROOT, category, uuid, "model.obj")
    if not os.path.exists(model_file):
        print("Model %s: file not exists!" % idx)
        print(model_file)
    else:
        print("Model %s, %s, %s" % (idx, category, uuid))

        # make output directory
        output_dir = os.path.join(RENDER_ROOT, idx)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # we do not do rendering in this script
        # instead, we write rendering scripts for each model
        # and use a synthesize script to call ImageMagick to synthesize image
        if platform.system() == "Windows":
            render_script = open(os.path.join(output_dir, "render.bat"), 'w')
            syn_script = open(os.path.join(output_dir, "synthesize.bat"), 'w')
        else: # assume linux or macos
            render_script = open(os.path.join(output_dir, "render.sh"), 'w')
            syn_script = open(os.path.join(output_dir, "synthesize.sh"), 'w')

        # random specular properties
        ks = random.random() * 0.2
        ns = random.random() * 1000
        # diffuse term
        kd = 1.0 - ks

        for env_file in envmaps:
            output_name = os.path.splitext(os.path.basename(env_file))[0]

            # generate a random viewpoint
            view = RandomView()

            # varibles for rendering
            var = "-Dmodel=\"%s\" -Denv=\"%s\" -Dview=%s -Dkd=%f -Dks=%f -Dns=%f" % (model_file, env_file, view, kd, ks, ns)

            # a base command

            # render command for image, albedo, shading, specular and depth
            # we do not need to directly render image since we can synthesize it by I = A*S + R
            #script.write("%s %s %s %s -o %s-i\n" % (MITSUBA, template_i, options, var, output_name))

            # rendering albedo
            render_script.write("%s %s %s %s -o %s_a\n" % (MITSUBA, template_a, options, var, output_name))

            # rendering shading
            render_script.write("%s %s %s %s -o %s_s\n" % (MITSUBA, template_s, options, var, output_name))

            # rendering specular
            render_script.write("%s %s %s %s -o %s_r\n" % (MITSUBA, template_r, options, var, output_name))

            # rendering depth, can be used for generate object mask
            render_script.write("%s %s %s %s -o %s_d\n" % (MITSUBA, template_d, options, var, output_name))


            # synthesize process script
            # we use .jpg LDR image instead of .exr HDR image to save disk space

            # first generate mask from depth
            syn_script.write("magick convert -format png -alpha off -depth 1 %s_d.exr %s_m.png\n" % (output_name,output_name))

            # albedo, shading and specular
            syn_script.write("magick mogrify -format jpg %s_a.exr\n" % output_name)
            syn_script.write("magick mogrify -format jpg %s_s.exr\n" % output_name)
            syn_script.write("magick mogrify -format jpg %s_r.exr\n" % output_name)

            # synthesize image
            syn_script.write("magick composite -compose Multiply %s_a.exr %s_s.exr temp.exr\n" % (output_name,output_name))
            syn_script.write("magick composite -compose Plus %s_r.exr temp.exr -format jpg %s_i.jpg\n" % (output_name,output_name))


        # clean
        if platform.system() == "Windows":
            syn_script.write("del temp.exr\n")
        else:
            syn_script.write("rm temp.exr\n")

        render_script.close()
        syn_script.close()

model_list.close()
