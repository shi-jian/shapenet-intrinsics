require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'

-- the network
function IntrinsicNetwork()

  local SConv = nn.SpatialConvolution
  local SBatchNorm = nn.SpatialBatchNormalization
  local SUpSamp = nn.SpatialUpSamplingBilinear

  -- input image
  local image = nn.Identity()():annotate{
   name = 'Input'
  } 

  -- encoder layers
  local conv0 = nn.Sequential()  
  conv0:add(SConv(3,16,3,3,1,1,1,1))
  conv0:add(SBatchNorm(16))
  conv0:add(nn.ReLU(true))       
  local conv00 = conv0(image) -- still 256x256
  
  local conv1 = nn.Sequential()
  conv1:add(SConv(16,32,3,3,2,2,1,1))
  conv1:add(SBatchNorm(32))
  conv1:add(nn.ReLU(true))
  local conv10 = conv1(conv00)  -- 128x128
  
  local conv2 = nn.Sequential()
  conv2:add(SConv(32,64,3,3,2,2,1,1))
  conv2:add(SBatchNorm(64))
  conv2:add(nn.ReLU(true))
  local conv20 = conv2(conv10) -- 64x64   
    
  local conv3 = nn.Sequential()
  conv3:add(SConv(64,128,3,3,2,2,1,1))
  conv3:add(SBatchNorm(128))
  conv3:add(nn.ReLU(true)) -- 32x32
  local conv30 = conv3(conv20)
  
  local conv4 = nn.Sequential()
  conv4:add(SConv(128,256,3,3,2,2,1,1))
  conv4:add(SBatchNorm(256))
  conv4:add(nn.ReLU(true)) -- 16x16
  local conv40 = conv4(conv30)
  
  local conv5 = nn.Sequential()
  conv5:add(SConv(256,256,3,3,2,2,1,1))
  conv5:add(SBatchNorm(256))
  conv5:add(nn.ReLU(true)) -- 8x8
  local conv50 = conv5(conv40)
  
  -- start decoder    
  local mid = {}
  local deconvs0 = {}
  local deconvs1 = {}
  local deconvs2 = {}
  local deconvs3 = {}
  local deconvs4 = {}
  local outputs = {}
  
  for i=1,3 do    
    local fc = nn.Sequential()
    fc:add(SConv(256,256,3,3,1,1,1,1))
    fc:add(SBatchNorm(256))
    fc:add(nn.ReLU(true)) -- 8x8
    fc:add(SConv(256,256,3,3,1,1,1,1))
    fc:add(SBatchNorm(256))
    fc:add(nn.ReLU(true)) -- 8x8
    fc:add(SConv(256,256,3,3,1,1,1,1))
    fc:add(SBatchNorm(256))
    fc:add(nn.ReLU(true)) -- 8x8
    fc:add(SConv(256,256,3,3,1,1,1,1))
    fc:add(SBatchNorm(256))
    fc:add(nn.ReLU(true)) -- 8x8
    mid[i] = fc(conv50)
  end
  mid[4] = conv50
  
  for i=1,3 do
    -- deconv and upsampling
    local deconv0 = nn.Sequential()
    deconv0:add(nn.JoinTable(2))
    deconv0:add(SConv(1024,256,3,3,1,1,1,1))
    deconv0:add(SBatchNorm(256))
    deconv0:add(nn.ReLU(true))
    deconv0:add(SUpSamp(2))
    deconvs0[i] = deconv0(mid) -- 16x16
  end
  deconvs0[4] = conv40
  
  for i=1,3 do
    local deconv1 = nn.Sequential()
    deconv1:add(nn.JoinTable(2))
    deconv1:add(SConv(1024,128,3,3,1,1,1,1))
    deconv1:add(SBatchNorm(128))
    deconv1:add(nn.ReLU(true))
    deconv1:add(SUpSamp(2))
    deconvs1[i] = deconv1(deconvs0) -- 32x32
  end
  deconvs1[4] = conv30
  
  for i=1,3 do        
    local deconv2 = nn.Sequential()
    deconv2:add(nn.JoinTable(2))
    deconv2:add(SConv(512,64,3,3,1,1,1,1))
    deconv2:add(SBatchNorm(64))
    deconv2:add(nn.ReLU(true))
    deconv2:add(SUpSamp(2))
    deconvs2[i] = deconv2(deconvs1) -- 64x64
  end
  deconvs2[4] = conv20
  
  for i=1,3 do    
    local deconv3 = nn.Sequential()
    deconv3:add(nn.JoinTable(2))
    deconv3:add(SConv(256,32,3,3,1,1,1,1))
    deconv3:add(SBatchNorm(32))
    deconv3:add(nn.ReLU(true))
    deconv3:add(SUpSamp(2))
    deconvs3[i] = deconv3(deconvs2) -- 128x128
  end
  deconvs3[4] = conv10
  
  for i=1,3 do    
    local deconv4 = nn.Sequential()
    deconv4:add(nn.JoinTable(2))
    deconv4:add(SConv(128,16,3,3,1,1,1,1))
    deconv4:add(SBatchNorm(16))
    deconv4:add(nn.ReLU(true))
    deconv4:add(SUpSamp(2))
    deconvs4[i] = deconv4(deconvs3) -- 256x256
  end
  deconvs4[4] = conv00  
  
  for i=1,3 do    
    -- output  
    local output4 = nn.Sequential()
    output4:add(nn.JoinTable(2))
    output4:add(SConv(64,16,3,3,1,1,1,1))
    output4:add(SBatchNorm(16))
    output4:add(nn.ReLU(true))
    
    -- image resolution
    output4:add(SConv(16,3,3,3,1,1,1,1))
    output4:add(SBatchNorm(3))
    output4:add(nn.ReLU(true))
    outputs[i] = output4(deconvs4) -- 3x256x256    
  end
  
  return nn.gModule({image}, outputs)
end

