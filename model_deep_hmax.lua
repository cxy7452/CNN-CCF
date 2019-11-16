--this CNN is based on Thomas Serre's hmax model

local function convKaiming(iDim, oDim, kW, kH, sW, sH)
	return cudnn.SpatialConvolution(iDim,oDim,kW,kH,sW,sH):init('weight',nninit.kaiming,{gain='relu'}):init('bias', nninit.constant, 0)
end

local function bnKaiming(oDim, bnW)
	return cudnn.SpatialBatchNormalization(oDim):init('weight', nninit.normal, 1.0, bnW):init('bias', nninit.constant,0)
end

function createModel(nGPU)
		
	-- actual net: ventral stream areas
	local v1 = nn.Sequential()
	v1:add(convKaiming(3,64,7,7,1,1))  -- RF = 1 degree = 7 pixels
	v1:add(bnKaiming(64,1e-3))
	v1:add(nn.ReLU(true))
	v1:add(nn.SpatialMaxPooling(3,3,2,2))
 
	local v2 = nn.Sequential()
	v2:add(convKaiming(64,96,3,3,1,1)) -- RF = 1.9 degree = 13 pixels
	v2:add(bnKaiming(96,1e-3))
	v2:add(nn.ReLU(true))
	
	local v4s = nn.Sequential() --takes the outputs of v2
	v4s:add(convKaiming(96,128,3,3,1,1)) -- RF = 2.4 degrees = 17 pixels
	v4s:add(bnKaiming(128,1e-3))
	v4s:add(nn.ReLU(true))
	v4s:add(nn.SpatialMaxPooling(3,3,2,2)) 	 	

	local v4c = nn.Sequential() -- takes the outputs of v4s
	v4c:add(convKaiming(128,192,4,4,1,1)) -- RF = 4.7 degrees = 33 pixels
	v4c:add(bnKaiming(192,1e-3))
	v4c:add(nn.ReLU(true))
	
	local teo_s1 = nn.Sequential() -- takes the outputs of v4c
	teo_s1:add(convKaiming(192,256,3,3,1,1)) -- RF = 5.9 degrees = 41 pixels
	teo_s1:add(bnKaiming(256,1e-3))
	teo_s1:add(nn.ReLU(true))
	teo_s1:add(nn.SpatialMaxPooling(3,3,2,2))

	local teo_s2 = nn.Sequential() -- takes the outputs of v2
	teo_s2:add(convKaiming(96,128,5,5,2,2)) -- Rf = 3 degrees = 21 pixels
	teo_s2:add(bnKaiming(128,1e-3))
	teo_s2:add(nn.ReLU(true))
	teo_s2:add(nn.SpatialAveragePooling(3,3,2,2))
	teo_s2:add(convKaiming(128,192,5,5,1,1)) -- RF = 7.6 degrees = 53 pixels
	teo_s2:add(bnKaiming(192,1e-3))
	teo_s2:add(nn.ReLU(true))
	teo_s2:add(nn.SpatialMaxPooling(2,2,1,1))

	local teo_c = nn.Sequential() -- takes the outputs of teo_s2
	teo_c:add(convKaiming(192,256,2,2,1,1)) -- Rf = 9.9 degrees = 69 pixels
	teo_c:add(bnKaiming(256,1e-3))
	teo_c:add(nn.ReLU(true))
	
	local te1 = nn.Sequential() -- takes input from teo_s1
	te1:add(convKaiming(256,192,5,5,1,1)) -- RF = 11.6 degrees = 81 pixels
	te1:add(bnKaiming(192,1e-3))
	te1:add(nn.ReLU(true))

	local te2 = nn.Sequential() -- takes input from v4c, te1, and teo_c
	te2:add(convKaiming(192+192+256,512,1,1,1,1)) -- dimensionality reduction
	te2:add(bnKaiming(512,1e-3))
	te2:add(nn.ReLU(true))
	te2:add(convKaiming(512,256,3,3,1,1))
	te2:add(bnKaiming(256,1e-3))
	te2:add(nn.ReLU(true))
	te2:add(nn.SpatialMaxPooling(3,3,2,2))

	local net = nn.Sequential()
	net:add(v1)
	net:add(v2)

	-- define the skip connections here
	local v4stream = nn.Sequential()
	local teo_s1stream = nn.Sequential()
	local teo_s2stream = nn.Sequential()
	
	local skip1 = nn.ConcatTable()
	local skip2 = nn.ConcatTable()
	
	skip2:add(teo_s1stream:add(teo_s1):add(te1))
	skip2:add(nn.SpatialAdaptiveMaxPooling(18,18))
	

	skip1:add(v4stream:add(v4s):add(v4c):add(skip2):add(nn.JoinTable(2)))
	skip1:add(teo_s2stream:add(teo_s2):add(teo_c):add(nn.SpatialAdaptiveMaxPooling(18,18)))		   	
	net:add(skip1):add(nn.JoinTable(2))
	net:add(te2)
	net:add(nn.View(256*7*7))

	-- FC components
	net:add(nn.Dropout(0.5))
    net:add(nn.Linear(256*7*7, 4096))
    net:add(cudnn.BatchNormalization(4096):init('weight', nninit.normal, 1.0, 0.001):init('bias', nninit.constant,0))
    net:add(nn.ReLU())

   	net:add(nn.Dropout(0.5))
   	net:add(nn.Linear(4096, 4096))
   	net:add(cudnn.BatchNormalization(4096):init('weight', nninit.normal, 1.0, 0.001):init('bias', nninit.constant,0))
   	net:add(nn.ReLU())

   	net:add(nn.Linear(4096, 1000))
   	net:add(nn.LogSoftMax())

	net:cuda()

	return net
end












