require 'torch'
require 'paths'
require 'image'
require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'

local cmd = torch.CmdLine()

cmd:option('-input', '', 'input image')
cmd:option('-mask', '', 'input mask')
cmd:option('-model', '', 'model file')
cmd:option('-outdir', '.', 'output directory')
cmd:option('-gpu', 1, 'use GPU')

local options = cmd:parse(arg)

local input = torch.FloatTensor(1, 3, 256, 256)
local mask  = torch.FloatTensor(1, 3, 256, 256)

-- load model
local model = torch.load(options.model)

-- load input image and mask
input[{1, {}, {}, {}}] = image.scale(image.load(options.input, 3), 256, 256)
mask[{1, {}, {}, {}}]  = image.scale(image.load(options.mask,  3), 256, 256)

if options.gpu == 0 then
  model:float()
else  
  model:cuda()
  input = input:cuda()
  mask = mask:cuda()
end

local pred = model:forward(input)

-- save output
image.save(paths.concat(options.outdir, 'albedo.png'), pred[1]:cmul(mask):squeeze())
image.save(paths.concat(options.outdir, 'shading.png'), pred[2]:cmul(mask):squeeze())
image.save(paths.concat(options.outdir, 'specular.png'), pred[3]:cmul(mask):squeeze())
-- save a copy of input
image.save(paths.concat(options.outdir, 'input.png'), input:squeeze())
image.save(paths.concat(options.outdir, 'mask.png'), mask:squeeze())

--EOF

