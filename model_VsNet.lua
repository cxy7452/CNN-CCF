--this CNN is based on our ventral stream model
-- aka VsNet

local function oneDConv(inDim, outDim)
	local oneD = nn.Sequential()
	oneD:add(cudnn.SpatialConvolution(inDim,outDim,1,1,1,1):init('weight',nninit.kaiming,{gain='relu'}):init('bias', nninit.constant, 0))
	oneD:add(cudnn.SpatialBatchNormalization(outDim):init('weight', nninit.normal, 1.0, 0.001):init('bias', nninit.constant,0))
	oneD:add(nn.ReLU(true))
	return oneD
end

local function maxPool(kS, sS)
	return nn.SpatialMaxPooling(kS,kS,sS,sS)
end

local function avgPool(kS, sS)
	return nn.SpatialAveragePooling(kS,kS,sS,sS)
end

local function convKaiming(iDim, oDim, kW, kH, sW, sH, pW, pH)
	return cudnn.SpatialConvolution(iDim,oDim,kW,kH,sW,sH,pW,pH):init('weight',nninit.kaiming,{gain='relu'}):init('bias', nninit.constant, 0)
end

local function bnKaiming(oDim, bnW)
	return cudnn.SpatialBatchNormalization(oDim):init('weight', nninit.normal, 1.0, bnW):init('bias', nninit.constant,0)
end

function createModel(nGPU)
		
	-- actual net: ventral stream areas
	local net = nn.Sequential()
 
	local v1 = nn.Sequential() -- input size is 224x224 ***************** v1 *****************
	local v1cat = nn.DepthConcat(2)
	local v1a = nn.Sequential():add(convKaiming(3,24,3,3,2,2,1,1))-- RF = 0.6 degree = 3 pixels
	v1a:add(bnKaiming(24,1e-3)):add(nn.ReLU(true))

	local v1b = nn.Sequential():add(convKaiming(3,22,5,5,2,2,2,2)) -- RF = 1.0 degree = 5 pixels
	v1b:add(bnKaiming(22,1e-3)):add(nn.ReLU(true))

	local v1c = nn.Sequential():add(convKaiming(3,18,7,7,2,2,3,3)) -- RF = 1.4 degree = 7 pixels
	v1c:add(bnKaiming(18,1e-3)):add(nn.ReLU(true))

	v1cat:add(v1a):add(v1b):add(v1c)
	v1:add(v1cat) -- output is 64x112x112


	local v2 = nn.Sequential() -- input size is 112x112 ***************** v2 *****************
	local v2cat = nn.DepthConcat(2)
	local v2a = oneDConv(64,30):add(convKaiming(30,30,3,3,1,1,1,1))-- RF = 1.4~2.2 degree
	v2a:add(bnKaiming(30,1e-3)):add(nn.ReLU(true))

	local v2b = oneDConv(64,28):add(convKaiming(28,28,5,5,1,1,2,2)) -- RF = 2.2~3.0 degree
	v2b:add(bnKaiming(28,1e-3)):add(nn.ReLU(true))

	local v2c = oneDConv(64,24):add(convKaiming(24,24,7,7,1,1,3,3)) -- RF = 3.0~3.8 degree
	v2c:add(bnKaiming(24,1e-3)):add(nn.ReLU(true))

	v2cat:add(v2a):add(v2b):add(v2c)
	v2:add(v2cat):add(maxPool(3,2))	 


	local v4 = nn.Sequential() -- input size is 55x55 ***************** v4 *****************
	local v4cat = nn.DepthConcat(2)
	local v4a = oneDConv(114,60):add(convKaiming(60,60,3,3,1,1,1,1))-- RF = 2.2~4.6 degree
	v4a:add(bnKaiming(60,1e-3)):add(nn.ReLU(true))

	local v4b = oneDConv(114,50):add(convKaiming(50,50,5,5,1,1,2,2)) -- RF = 3.0~5.4 degree
	v4b:add(bnKaiming(50,1e-3)):add(nn.ReLU(true))

	v4cat:add(v4a):add(v4b)
	v4:add(v4cat) 


	local teo = nn.Sequential() -- input size is 55x55 ***************** TEO *****************
	local teocat = nn.DepthConcat(2)
	local teoa = oneDConv(192,100):add(convKaiming(100,100,3,3,1,1,1,1))-- RF = 6.2~8.6 degree
	teoa:add(bnKaiming(100,1e-3)):add(nn.ReLU(true))

	local teob = oneDConv(192,100):add(convKaiming(100,100,5,5,1,1,2,2)) -- RF = 7.8~10.2 degree
	teob:add(bnKaiming(100,1e-3)):add(nn.ReLU(true))

	teocat:add(teoa):add(teob):add(teoc)
	teo:add(teocat):add(maxPool(3,2)) 


	local te = nn.Sequential() -- input size is 27x27 ***************** TE *****************
	local tecat = nn.DepthConcat(2)
	local tea = oneDConv(310,91):add(convKaiming(91,91,3,3,1,1,1,1))-- RF = 11~16.6 degree
	tea:add(bnKaiming(91,1e-3)):add(nn.ReLU(true))

	local teb = oneDConv(310,91):add(convKaiming(91,91,5,5,1,1,2,2)) -- RF = 14.2~19.8 degree
	teb:add(bnKaiming(91,1e-3)):add(nn.ReLU(true))

	local tec = oneDConv(310,91):add(convKaiming(91,91,7,7,1,1,3,3)) -- RF = 19~24.6 degree
	tec:add(bnKaiming(91,1e-3)):add(nn.ReLU(true))

	tecat:add(tea):add(teb):add(tec)
	--te:add(tecat):add(oneDConv(333,256)):add(maxPool(3,2)) -- output is 256x13x13
	te:add(tecat):add(maxPool(5,4)) 

	-- bypass connection implementations
	local v1split = nn.ConcatTable()
	v1split:add(nn.Sequential():add(oneDConv(64,32)):add(avgPool(3,2))) -- weak bypass connection to v4
	v1split:add(v2)
	v1:add(v1split)

	local v2stream = nn.ConcatTable()
	v2stream:add(nn.Identity())  
	v2stream:add(nn.Identity())
	v2:add(v2stream)	

	local v4stream = nn.ConcatTable()
	v4stream:add(nn.Identity())
	v4stream:add(nn.Identity())
	
	local v124 = nn.ConcatTable()
	local v24 = nn.Sequential():add(nn.NarrowTable(1,2)):add(nn.JoinTable(2)):add(v4):add(v4stream)
	v124:add(v24) -- combining v1 and v2 into v4
	v124:add(nn.SelectTable(3))  -- strong bypass connection to TEO

	local v24t = nn.ConcatTable()
	local v4t = nn.Sequential():add(nn.NarrowTable(2,2)):add(nn.JoinTable(2)):add(teo)
	v24t:add(nn.Sequential():add(nn.SelectTable(1)):add(avgPool(3,2)))
	v24t:add(v4t)

	net:add(v1):add(nn.FlattenTable()):add(v124):add(nn.FlattenTable()):add(v24t):add(nn.JoinTable(2)):add(te)--:add(avgPool(3,2))

	net:add(nn.View(273*6*6))

	-- FC components
	net:add(nn.Dropout(0.5))
    net:add(nn.Linear(273*6*6, 4096))
    net:add(cudnn.BatchNormalization(4096):init('weight', nninit.normal, 1.0, 0.001):init('bias', nninit.constant,0))
    net:add(nn.ReLU())

   	net:add(nn.Dropout(0.5))
   	net:add(nn.Linear(4096, 4096))
   	net:add(cudnn.BatchNormalization(4096):init('weight', nninit.normal, 1.0, 0.001):init('bias', nninit.constant,0))
   	net:add(nn.ReLU())

   	net:add(nn.Linear(4096, 1000))
   	net:add(nn.LogSoftMax())

	net:cuda()
	cudnn.convert(net,cudnn)
	return net
end












