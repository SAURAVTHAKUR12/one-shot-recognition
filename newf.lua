require 'nn'
local net = nn.Sequential();
local inputs=2;local outputs=1;local HUs=20;
net:add(nn.Linear(inputs,HUs))
net:add(nn.Tanh())
net:add(nn.Linear(HUs,outputs))
local criterion=nn.MSECriterion()
local batchSize=128
local batchInputs=torch.Tensor(batchSize,inputs)
local batchLabels=torch.ByteTensor(batchSize)
for i=1,batchSize do
local input=torch.randn(2)
local label=1
if input[1]*input[2]>0 then
label=-1;
end
batchInputs[i]:copy(input)
batchLabels[i]=label
end
local params, gradParams=net:getParameters()
local optimState={learningRate=0.01}
require 'optim'
for epoch=1,50 do
local function feval(params)
gradParams:zero()
local outputs=net:forward(batchInputs)
local loss=criterion:forward(outputs,batchLabels)
local dloss_doutput=criterion:backward(outputs,batchLabels)
net:backward(batchInputs,dloss_doutput)
return loss,gradParams
end
optim.sgd(feval,params,optimState)
end

x=torch.Tensor(2)
x[1]=0.5;
x[2]=0.5;
print(net:forward(x))
print('the next is \n')
x[1]=0.5;
x[2]=-0.5;
print(net:forward(x))
print('the next is \n')
x[1]=-0.5;
x[2]=0.5;
print(net:forward(x))
 
