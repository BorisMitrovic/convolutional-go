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

% args 
batchSize = 100;

for tr=0:1,
set = '';
if tr,
  set = 'train';
else
  set = 'test';
end
    
nData =0;
nPerClass = [];
for i =1:nClasses,
  setinst = sprintf('%s%d', set, i-1);
  load(setinst); 
  nData = nData + length(D);
  nPerClass = [nPerClass, length(D)];
end

fprintf(1, 'Size of the %sing dataset= %5d \n',set, nData);




if tr,
  nTrData = nData;
  nTrPerClass = nPerClass;
else
  nTsData = nData;
  nTsPerClass = nPerClass;
end

% return
nData = [nTsData; nTrData];
nPerClass = [nTsPerClass; nTrPerClass];
endPerClass = cumsum(nPerClass,2);
nBatches = nData / batchSize;
nDims = size(D,2);

clear D;
end

%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 



