require 'nn'

local SpatialUpSamplingBilinear = nn.SpatialUpSamplingBilinear

function SpatialUpSamplingBilinear:setSize(input)
   local xdim = input:dim()
   local ydim = xdim - 1
   for i = 1, input:dim() do
      self.inputSize[i] = input:size(i)
      self.outputSize[i] = input:size(i)
   end
   if self.scale_factor ~= nil then
      self.outputSize[ydim] = (self.outputSize[ydim]-1) * (self.scale_factor-1)
                               + self.outputSize[ydim]
      self.outputSize[xdim] = (self.outputSize[xdim]-1) * (self.scale_factor -1)
                               + self.outputSize[xdim]
   else
      self.outputSize[ydim] = self.oheight
      self.outputSize[xdim] = self.owidth
   end
end

