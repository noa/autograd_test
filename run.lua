#! /usr/bin/env th

-- This is small modification to the train-penn example:
-- the lstm layers take an extra discrete input.
--
-- So a batch consists of input, target, and label where:
--   * label is a vector of size input:size(1)
--   * a oneHot embedding is used for the labels

local opt = {}
opt.type = 'double' -- cuda | double
opt.gpu = 1
opt.paramRange = 0.1
opt.embedDim = 64
opt.hiddenDim = 512
opt.reportEvery = 100
opt.nEpochs = 10
opt.reportEvery = 100
opt.learningRate = 0.1
opt.dropout = 0.25
opt.maxGradNorm = 0.25

-- Seed
torch.manualSeed(42)

if opt.type == 'cuda' then
   require 'cutorch'
   require 'cunn'
   assert(type(opt.gpu) == 'number')
   cutorch.setDevice(opt.gpu)
   local freeMemory, totalMemory = cutorch.getMemoryUsage(opt.gpu)
   print("GPU ("..opt.gpu..") free = "..math.floor(freeMemory*1.0e-9).."G"
            .." total = "..math.floor(totalMemory*1.0e-9).."G")
   cutorch.manualSeed(42)
end

local function get_batches(path)
   local ret = torch.load(path, 'ascii')
   if opt.type == 'cuda' then
      for k,batch in pairs(ret) do
         for kk,_ in pairs(batch) do
            batch[kk] = batch[kk]:cuda()
         end
      end
   end
   return ret
end

-- Read serialized batches
local metadata = torch.load('data/meta.ser', 'ascii')

local inputSize = assert(metadata.inputSize)
local labelSize = assert(metadata.labelSize)

local train_batches = get_batches('data/train.ser')
local valid_batches = get_batches('data/valid.ser')
local test_batches = get_batches('data/test.ser')

-- Autograd Libs
local autograd = require 'autograd'
local util = require 'autograd.util'
local moses = require 'moses'
local RecurrentLSTM = require 'lstm'

-- Get model parameters
local params = {}

-- Define LSTM layers:
local lstm1,params = RecurrentLSTM({
      inputFeatures = opt.embedDim,
      labelFeatures = labelSize,
      hiddenFeatures = opt.hiddenDim,
      outputType = 'all'})

local lstm2 = RecurrentLSTM({
      inputFeatures = opt.hiddenDim,
      labelFeatures = labelSize,
      hiddenFeatures = opt.hiddenDim,
      outputType = 'all'}, params)

local lstm3 = RecurrentLSTM({
      inputFeatures = opt.hiddenDim,
      labelFeatures = labelSize,
      hiddenFeatures = opt.hiddenDim,
      outputType = 'all'}, params)

-- oneHot is broken in latest torch-autograd commit
local function oneHot(labels, n)
   local n = n or torch.max(labels)
   local nLabels = torch.size(labels, 1)
   local out = labels.new(nLabels, n)
   torch.fill(out, 0)
   for i=1,nLabels do
      out[i][labels[i]] = 1.0
   end
   return out
end

-- Dropout
local regularize = util.dropout

-- Use built-in nn modules:
local lsm = autograd.nn.LogSoftMax()
local lossf = autograd.nn.ClassNLLCriterion()

-- Complete trainable function:
local f = function(params, input, target, label, prevState, dropout)
   -- N elements:
   local batchSize = torch.size(input, 1)
   local bpropLength = torch.size(input, 2)
   local nElements = batchSize * bpropLength

   -- Select word vectors
   local x = util.lookup(params.words.W, input)
   local l = oneHot(label, labelSize)

   -- Encode all inputs through LSTM layers:
   local h1,newState1 = lstm1(params[1], regularize(x,dropout), l, prevState[1])
   local h2,newState2 = lstm2(params[2], regularize(h1,dropout), l, prevState[2])
   local h3,newState3 = lstm3(params[3], regularize(h2,dropout), l, prevState[3])

   -- Flatten batch + temporal
   local h3f = torch.view(h3, nElements, opt.hiddenDim)
   local yf = torch.view(target, nElements)

   -- Linear classifier:
   local h4 = regularize(h3f,dropout) * params[4].W + torch.expand(params[4].b,
                                                                   nElements,
                                                                   inputSize)

   -- Lsm
   local yhat = lsm(h4)

   -- Loss:
   local loss = lossf(yhat, yf)

   -- Return avergage loss
   return loss, {newState1, newState2, newState3}
end

-- Linear classifier params:
table.insert(params, {
   W = torch.Tensor(opt.hiddenDim, inputSize),
   b = torch.Tensor(1, inputSize),
})

-- Init weights + cast:
for i,weights in ipairs(params) do
   for k,weight in pairs(weights) do
      if opt.type == 'cuda' then
         weights[k] = weights[k]:cuda()
      elseif opt.type == 'double' then
         weights[k] = weights[k]:double()
      else
         weights[k] = weights[k]:float()
      end
      weights[k]:uniform(-opt.paramRange, opt.paramRange)
   end
end

-- Word dictionary to train:
local words
if opt.type == 'cuda' then
   words = torch.CudaTensor(inputSize, opt.embedDim)
elseif opt.type == 'double' then
   words = torch.DoubleTensor(inputSize, opt.embedDim)
else
   words = torch.FloatTensor(inputSize, opt.embedDim)
end
words:uniform(-opt.paramRange, opt.paramRange)
params.words = {W = words}

-- Get the computation graph
local df = autograd(f)

-- Compute perplexity
function compute_perplexity(batches)
   local n_batch = #batches
   local aloss = 0
   local loss
   local lstmState
   for batch_idx = 1, n_batch do
      local batch = batches[batch_idx]
      loss, lstmState = f(params,
                          batch.inputs,
                          batch.targets,
                          batch.labels,
                          {},
                          0) -- no dropout
      aloss = aloss + loss
      collectgarbage()
   end
   aloss = aloss / n_batch
   return math.exp(aloss)
end

-- Do training
local n_batch = #train_batches
local aloss = 0
local loss
local lstmState
local grads
local lr = assert(opt.learningRate)
local reportEvery = assert(opt.reportEvery)
local learningRate = assert(opt.learningRate)
local valPerplexity = math.huge

for epoch = 1, opt.nEpochs do

   -- Train:
   print('\nTraining Epoch #'..epoch)

   local aloss = 0
   local maxGrad = 0
   local lstmState
   local grads,loss

   -- Train:
   for batch_idx = 1, n_batch do
      xlua.progress(batch_idx, n_batch)

      -- get the next batch
      local batch = train_batches[batch_idx]

      -- Grads:
      grads,loss,lstmState = df(params,
                                batch.inputs,
                                batch.targets,
                                batch.labels,
                                {},
                                opt.dropout)


      -- Cap gradient norms:
      local norm = 0
      for i,grad in ipairs(moses.flatten(grads)) do
         norm = norm + torch.sum(torch.pow(grad,2))
      end
      norm = math.sqrt(norm)
      if norm > opt.maxGradNorm then
         for i,grad in ipairs(moses.flatten(grads)) do
            grad:mul( opt.maxGradNorm / norm )
         end
      end

      -- Update params:
      for k,vs in pairs(grads) do
         for kk,v in pairs(vs) do
            params[k][kk]:add(-lr, grads[k][kk])
         end
      end

      -- Loss: exponentiate nll gives perplexity
      aloss = aloss + loss
      if batch_idx % reportEvery == 0 then
         aloss = aloss / reportEvery
         local perp = math.exp(aloss)
         print('\nAverage training perplexity = ' .. perp)
         aloss = 0
      end

      collectgarbage()
   end

   -- Valid:
   local newValPerplexity = compute_perplexity(valid_batches)
   print('Validation perplexity = ' .. newValPerplexity)

   -- Learning rate scheme:
   if newValPerplexity > valPerplexity or
   (valPerplexity - newValPerplexity)/valPerplexity < .10 then
      -- No progress made, decrease learning rate
      learningRate = learningRate / 2
      print('Validation perplexity stagnating, decreasing learning rate to: ' .. learningRate)
   end
   valPerplexity = newValPerplexity

   -- Test:
   print('\nTest set [just indicative, not used for training]...')
   print('Test set perplexity = ' .. compute_perplexity(test_batches))

   collectgarbage()
end
