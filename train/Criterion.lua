require 'torch'
require 'nn'
require 'cunn'

-- weighted MSE criterion, scale invariant, and with mask
local WeightedMSE, parent = torch.class('nn.WeightedMSE', 'nn.Criterion')

function WeightedMSE:__init(scale_invariant)
  parent.__init(self)    
  -- we use a standard MSE criterion internally
  self.criterion = nn.MSECriterion()
  self.criterion.sizeAverage = false
  
  -- whether consider scale invarient
  self.scale_invariant = scale_invariant or false
end

-- targets should contains {target, weight}
function WeightedMSE:updateOutput(pred, targets)

  local target = targets[1]
  local weight = targets[2]
  
  -- scale-invariant: rescale the pred to target scale
  if self.scale_invariant then
  
    -- get the dimension and size
    local dim = target:dim()
    local size = target:size()
    for i=1,dim-2 do
      size[i] = 1
    end
  
    -- scale invariant
    local tensor1 = torch.cmul(pred, target)
    local tensor2 = torch.cmul(pred, pred)
    
    -- get the scale
    self.scale = torch.cdiv(tensor1:sum(dim):sum(dim-1),tensor2:sum(dim):sum(dim-1))
    -- patch NaN
    self.scale[self.scale:ne(self.scale)] = 1
    
    -- constrain the scale in [0.1, 10]
    self.scale:cmin(10)
    self.scale:cmax(0.1)
        
    -- expand the scale 
    self.scale = self.scale:repeatTensor(size)
   
    --  re-scale the pred   
    pred:cmul(self.scale)
  end
  
  -- sum for normalize
  self.alpha = torch.cmul(weight, weight):sum()
  if self.alpha ~= 0 then
    self.alpha = 1 / self.alpha
  end
  
  -- apply weight to pred and target, and keep a record for them so that we do not need to re-calculate
  self.weighted_pred    = torch.cmul(pred, weight)
  self.weighted_target  = torch.cmul(target, weight)
        
  return self.criterion:forward(self.weighted_pred, self.weighted_target) * self.alpha
end

function WeightedMSE:updateGradInput(input, target) 

  self.grad = self.criterion:backward(self.weighted_pred, self.weighted_target)
  
  if self.scale then
    self.grad:cdiv(self.scale)
    -- patch NaN
    self.grad[self.grad:ne(self.grad)] = 0
  end
  
  return self.grad * self.alpha
  
end

function WeightedMSE:cuda()
  self.criterion:cuda()
end


-- build convolutional kernel for calculate image gradient
local x = nn.SpatialConvolution(3,3,3,3,1,1,1,1)
x.weight:zero()
x.bias:zero()

local y = nn.SpatialConvolution(3,3,3,3,1,1,1,1)
y.weight:zero()
y.bias:zero()  

for i = 1, 3 do
  x.weight[{i, i, {}, {}}] = torch.Tensor({{-1, 0, 1},{-2, 0, 2},{-1, 0, 1}})
  y.weight[{i, i, {}, {}}] = torch.Tensor({{-1, -2, -1},{0, 0, 0},{1, 2, 1}})
end

-- gradient
local function Gradient(image)  
  local ix = x:forward(image)
  local iy = y:forward(image)  
  return torch.sqrt(torch.add(torch.cmul(ix, ix), torch.cmul(iy, iy)))
end

-- criterion for multiple output(albedo/shading/specular)
IntrinsicCriterion, parent = torch.class('nn.IntrinsicCriterion', 'nn.Criterion')

function IntrinsicCriterion:__init() 
  
  -- criterions
  self.criterion_a = nn.MultiCriterion()  
  self.criterion_a:add(nn.WeightedMSE(true), 0.95) 
  self.criterion_a:add(nn.WeightedMSE(false), 0.05)
  
  -- for shading
  self.criterion_s = nn.MultiCriterion()  
  self.criterion_s:add(nn.WeightedMSE(true), 0.95) 
  self.criterion_s:add(nn.WeightedMSE(false), 0.05)
    
  -- for specular
  self.criterion_r = nn.MultiCriterion()  
  self.criterion_r:add(nn.WeightedMSE(false), 1)

end

-- pred contains a table of {pred_albedo, pred_shading, pred_specular}
-- target contains a table of {input, mask, albedo, shading, specular}
function IntrinsicCriterion:updateOutput(pred, target)
  -- input might be useful to calculate weight
  local input = target[1]
  local mask  = target[2] 
  
  -- we can mask out background pixel of prediction here
  local gt_a  = torch.cmul(mask, target[3])
  local gt_s  = torch.cmul(mask, target[4])
  local gt_r  = torch.cmul(mask, target[5])  
  
  local pd_a = torch.cmul(mask, pred[1])
  local pd_s = torch.cmul(mask, pred[2])
  local pd_r = torch.cmul(mask, pred[3])
  
  -- calculate weight
  local weight
  local useGradient = false
  if useGradient then
    local gradient = torch.exp(Gradient(input))
    weight = torch.cmul(mask, gradient)
  else    
    weight = mask
  end
     
  self.loss_a = self.criterion_a:forward(pd_a, {gt_a, weight})
  self.loss_s = self.criterion_s:forward(pd_s, {gt_s, weight})
  self.loss_r = self.criterion_r:forward(pd_r, {gt_r, weight})
    
  return {self.loss_a, self.loss_s, self.loss_r}
end

function IntrinsicCriterion:updateGradInput(pred, target)

  local input = target[1]
  local mask  = target[2] 
  local gt_a  = target[3]
  local gt_s  = target[4]
  local gt_r  = target[5]
  
  local pd_a = pred[1]
  local pd_s = pred[2]
  local pd_r = pred[3]
     
  self.grad_a = self.criterion_a:backward(pd_a, {gt_a, mask})
  self.grad_s = self.criterion_s:backward(pd_s, {gt_s, mask})
  self.grad_r = self.criterion_r:backward(pd_r, {gt_r, mask})
  
  return {self.grad_a, self.grad_s, self.grad_r}
end

function IntrinsicCriterion:cuda()
  -- convert to cuda   
  self.criterion_a:cuda() 
  self.criterion_s:cuda() 
  self.criterion_r:cuda()  
end







