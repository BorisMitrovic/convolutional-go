function [ batch,targets ] = getBatch( nBatch,tr,endPerClass )
%GETBATCH Summary of this function goes here
%   Detailed explanation goes here
  batch=[];
  targets=[];

  rand('state',0); %so we know the permutation of the training data
  randomorder=randperm(nData(tr));

  b = nBatch;
  for r=1+(b-1)*batchsize:b*batchsize,
    i= randomorder(r);
    [instance,target] = getSingle(i,endPerClass(tr));
    targets = [targets, target];
    batch   = [batch, instance];
  end
batch = batch/2;

end

