% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton
%
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our
% web page.
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.

data=[]; 
targets=[]; 
for tr=0:1,
set = '';
if tr,
  set = 'train';
else
  set = 'test';
end
    set
for i =1:nClasses,
  setinst = sprintf('%s%d', set, i-1);
  load(setinst); 
  data = [data; D];
  row = getTarget(i,nClasses);
  targets = [targets; repmat(row, size(D,1), 1)];  
end
data = sign(data);  % modelling presence / absence

totnum=size(data,1);
fprintf(1, 'Size of the %sing dataset= %5d \n',set, totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

numbatches=totnum/100;
numdims  =  size(data,2);
batchsize = 100;
  bdata = zeros([batchsize, numdims, numbatches]);
  btargets = zeros([batchsize, nClasses, numbatches]);
  for b=1:numbatches
    bdata(:,:,b) = data(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    btargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  end
if tr,
    batchdata = bdata;
    batchtargets = btargets;
else
  testbatchdata = bdata;
  testbatchtargets = btargets;
end

end
clear data targets bdata btargets;


%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 



