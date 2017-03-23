Dataset = {}
Dataset.__index = Dataset

-- a simple dataset class
function Dataset.load(list1, list2)
  local dataset = {}
  setmetatable(dataset, Dataset) 
 
  dataset.size1 = #list1
  dataset.size2 = #list2
  dataset.size = dataset.size1 * dataset.size2
  
  dataset.list1 = list1
  dataset.list2 = list2
  
  -- build a index table, we use tensor to get avoid gc
  dataset.idx = torch.LongStorage(dataset.size)
  for i=1,dataset.size do
    dataset.idx[i] = i
  end
    
  return dataset
end

function Dataset:shuffle(seed)
  math.randomseed(seed or 0)  
  for i=self.size, 2, -1 do
    -- get a random index for shuffle
    local j = math.random(i)    
    -- shuffle indices
    self.idx[i],self.idx[j] = self.idx[j],self.idx[i]
  end 
end

function Dataset:get(index)  
  -- keep index in [1, self.size]
  index = (index - 1) % self.size + 1
    
  local idx1 = (self.idx[index] - 1) % self.size1 + 1
  local idx2 = math.floor((self.idx[index] - 1) / self.size1 + 1)
  return self.list1[idx1], self.list2[idx2]
end



