require 'paths'
if (not paths.filep("cifar10torchsmall.zip")) then
os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
os.execute('unzip cifar10torchsmall.zip')
end
trainset=torch.load('cifar10-train.t7')
testset=torch.load('cifar10-test.t7')
classes={'airplane','automobile','bird','cat','deer','dog','frog','ship','truck'}
print(trainset)
print(#trainset.data)
itorch.image(trainset.data[100])
print(classes[trainset.label[100]])
