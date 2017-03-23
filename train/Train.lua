require 'torch'
require 'paths'
require 'image'

-- 
require 'Dataset'
require 'Network'
require 'Criterion'

-- parse command line parameters
local cmd = torch.CmdLine()

cmd:option('-data_root',  os.getenv('RENDER_ROOT') or '', 'the dataset root directory')
cmd:option('-model_list', paths.concat(os.getenv('RENDER_ROOT') or '','dataset.csv'))
cmd:option('-env_list',   paths.concat(os.getenv('ENVMAP_ROOT') or '','list.txt'))
cmd:option('-outdir', '.', 'output directory')

cmd:option('-max_iter', 1000000, 'max training iteration')
cmd:option('-save_iter', 10000, 'save snapshot')
cmd:option('-test_iter', 10000, 'test')
cmd:option('-disp_iter', 10, 'display error and training result')

cmd:option('-snapshot', '', 'snapshot file')

cmd:option('-batch_size', 4, 'batch size')
cmd:option('-cuda', true, 'use cuda')
cmd:option('-cudnn', true, 'use cudnn')
cmd:option('-devid', 1, 'cuda device index')
cmd:option('-seed', 666, 'random seed')

local options = cmd:parse(arg)

-- check input files
if not paths.filep(options.model_list) then
  print('Please check model_list file!')
  os.exit(1)
end

if not paths.filep(options.env_list) then
  print('Please check env_list file!')
  os.exit(1)
end

-- load ShapeNet model list
local model_train = {}
local model_test = {}
local model_val = {}

-- read model list
for line in io.lines(options.model_list) do
  local split = {}  
  for token in line:gmatch('([^,]+)') do
    table.insert(split, token)
  end
  
  -- split shapenet model for training
  if split[5] == 'train' then
    table.insert(model_train, split[1])
  elseif split[5] == 'test' then
    table.insert(model_test, split[1])
  elseif split[5] == 'val' then
    table.insert(model_val, split[1])
  end
end   

-- print(#model_train, #model_test, #model_val)

-- load environment map list
local env_train = {}
local env_test = {}
local env_val = {}

for line in io.lines(options.env_list) do
  local split = {}  
  for token in paths.basename(line):gmatch('([^.]+)') do
    table.insert(split, token)
  end
  
  -- we use all environment maps for both training and testing
  -- since we have relative small size environment map dataset, it is not a good idea to random split it.
  -- experiments showed that reasonably split environment maps would produce results very close to no-split setting
  table.insert(env_train, split[1])
  table.insert(env_test, split[1])
  table.insert(env_val, split[1])  
end

-- create training dataset
local dataset = Dataset.load(model_train, env_train)
dataset:shuffle()
print('Training dataset size:', dataset.size)

-- pre-allocate memory, hard code for image resolution
local input = torch.Tensor(options.batch_size, 3, 256, 256)
local mask = torch.Tensor(options.batch_size, 3, 256, 256)
local target_albedo = torch.Tensor(options.batch_size, 3, 256, 256)
local target_shading = torch.Tensor(options.batch_size, 3, 256, 256)
local target_specular = torch.Tensor(options.batch_size, 3, 256, 256)

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(options.seed)
if options.cuda then  
  cutorch.setDevice(options.devid)
  cutorch.manualSeed(options.seed)
end

-- network and criterion
local network

if options.snapshot and paths.filep(options.snapshot) then
  -- load from snapshot
  print('Load snapshot file...')
  network = torch.load(options.snapshot)
else
  network = IntrinsicNetwork()
end

local criterion = nn.IntrinsicCriterion()

-- cuda
if options.cuda then
  require 'cunn'
  
  input = input:cuda()
  mask = mask:cuda()
  target_albedo = target_albedo:cuda()
  target_shading = target_shading:cuda()
  target_specular = target_specular:cuda()

  network:cuda()
  criterion:cuda()
  
  if options.cudnn then
    require 'cudnn'
    --cudnn.fastest = true
    cudnn.convert(network, cudnn)
  end  
end

-- for solver
local x, dl_dx = network:getParameters()

local optim = require('optim')
--local solver = optim['adam']
local solver = optim['adadelta']
local state = {
  learningRate = 0.01,
  learningRateDecay = 1e-5
}

-- for display images and error
local display = require 'display'

-- timer
local timer = torch.Timer()
timer:reset()

local loss_albedo = 0
local loss_shading = 0
local loss_specular = 0

local loss_table_train = {}
local loss_table_test = {}

local plot_config_train = {
  title = "Training Loss",
  labels = {"iter", "albedo", "shading", "specular"},
  ylabel = "Weighted MSE",
}

local plot_config_test = {
  title = "Testing Loss",
  labels = {"iter", "albedo", "shading", "specular"},
  ylabel = "Weighted MSE",
}

local iter = 0
local curr = 0 -- current sample index
while iter < options.max_iter do
  iter = iter + 1
  
  network:training()
    
  for i=1,options.batch_size do
    curr = curr + 1
    local model, env = dataset:get(curr)
    local prefix = paths.concat(options.data_root, model, env)
    
    -- load data
    input[{i,{},{},{}}] = image.load(prefix..'_i.jpg', 3)
    mask[{i,{},{},{}}] = image.load(prefix..'_m.png', 3)
    
    target_albedo[{i,{},{},{}}] = image.load(prefix..'_a.jpg', 3)
    target_shading[{i,{},{},{}}] = image.load(prefix..'_s.jpg', 3)
    target_specular[{i,{},{},{}}] = image.load(prefix..'_r.jpg', 3)    
  end
  
  -- mask out input background?
  -- input:cmul(mask)
  
  -- forward
  local pred = network:forward(input)
  local pred_albedo = pred[1]:cmul(mask)
  local pred_shading = pred[2]:cmul(mask)
  local pred_specular = pred[3]:cmul(mask)
  
  -- get loss
  local loss = criterion:forward(pred, {input, mask, target_albedo, target_shading, target_specular})
  loss_albedo = loss_albedo + loss[1]
  loss_shading = loss_shading + loss[2]
  loss_specular = loss_specular + loss[3]
  
  -- get gradient
  local grad = criterion:backward(pred, {input, mask, target_albedo, target_shading, target_specular})
  
  -- optimize
  local function feval()    
    -- update parameters
    network:zeroGradParameters()
    network:backward(input,grad)        
    return loss, dl_dx
  end
  solver(feval, x, state)  
  
  
  if iter % options.disp_iter == 0 then
    -- display training images and plot error
    win_image = display.image(input, {win=win_image,title='Input Image, iter:'..iter})    
    win_a0 = display.image(torch.cat(target_albedo,pred_albedo, 4), {win=win_a0,title='Albedo, iter:'..iter})    
    win_s0 = display.image(torch.cat(target_shading,pred_shading, 4), {win=win_s0,title='Shading, iter:'..iter})    
    win_r0 = display.image(torch.cat(target_specular,pred_specular, 4), {win=win_r0,title='Specular, iter:'..iter})

    loss_albedo = loss_albedo / options.disp_iter
    loss_shading = loss_shading / options.disp_iter
    loss_specular = loss_specular / options.disp_iter
    
    table.insert(loss_table_train, {iter, loss_albedo, loss_shading, loss_specular})
    plot_config_train.win = display.plot(loss_table_train, plot_config_train)
    
    -- print to console
    print(string.format("Iteration %d, %d/%d samples, %.2f seconds passed",
                        iter, curr, dataset.size, timer:time().real))
    print(string.format("\t Loss-a: %.4f, Loss-s: %.4f, Loss-r: %.4f",
                        loss_albedo, loss_shading, loss_specular))
    
    loss_albedo = 0
    loss_shading = 0
    loss_specular = 0 
  end
  
  if iter % options.test_iter == 0 then
    -- testing...
    network:evaluate()
    
    local test_loss_a = 0
    local test_loss_s = 0
    local test_loss_r = 0
    
    for k = 1,#model_test,options.batch_size do
    
      for i=1, options.batch_size do
        local idx0 = (k + i - 2) % #model_test + 1
        local idx1 = (k + i - 2) % #env_test + 1
        
        -- go through models
        local model = model_test[idx0]
        -- we only use single envmap to save testing time
        local env = env_test[idx1]
        
        local prefix = paths.concat(options.data_root, model, env)
    
        -- load data
        input[{i,{},{},{}}] = image.load(prefix..'_i.jpg', 3)
        mask[{i,{},{},{}}] = image.load(prefix..'_m.png', 3)
        
        target_albedo[{i,{},{},{}}] = image.load(prefix..'_a.jpg', 3)
        target_shading[{i,{},{},{}}] = image.load(prefix..'_s.jpg', 3)
        target_specular[{i,{},{},{}}] = image.load(prefix..'_r.jpg', 3) 
      end
      
        -- forward
      local pred = network:forward(input)
      
      -- get loss
      local loss = criterion:forward(pred, {input, mask, target_albedo, target_shading, target_specular})
      test_loss_a = test_loss_a + loss[1]
      test_loss_s = test_loss_s + loss[2]
      test_loss_r = test_loss_r + loss[3] 
                    
    end
    
    test_loss_a = test_loss_a / #model_test
    test_loss_s = test_loss_s / #model_test
    test_loss_r = test_loss_r / #model_test
    
    print(string.format("Evaluation, Loss-a: %.4f, Loss-s: %.4f, Loss-r: %.4f",
                        test_loss_a, test_loss_s, test_loss_r))
                        
    table.insert(loss_table_test, {iter, test_loss_a, test_loss_s, test_loss_r})
    plot_config_test.win = display.plot(loss_table_test, plot_config_test)
  end
    
  if iter % options.save_iter == 0 then
    -- save model
    print("Save model on iteration", iter)    
    network:clearState()
    torch.save(paths.concat(options.outdir, 'snapshot_'..iter..'.t7'), network)
  end  

  -- manually GC
  collectgarbage() 
  
end




