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

% This program reads raw MNIST files available at 
% http://yann.lecun.com/exdb/mnist/ 
% and converts them to files in matlab format 
% Before using this program you first need to download files:
% train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz 
% t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz
% and gunzip them. You need to allocate some space for this.  

% This program was originally written by Yee Whye Teh 

% Modified by Boris Mitrovic

trainData = 'presence_train0.txt';
trainLbls = 'presence_train0.lbl';
testData  = 'presence_test0.txt';
testLbls  = 'presence_test0.lbl';
train = 'presence_train';
test = 'presence_test';
data = '.txt';
lbls = '.lbl';

nClasses = 361;
nTest  = 1903; 
nTrain = 1495;
sz = 19;


% Work with test files first 
fprintf(1,'Go data \n'); 


fprintf(1,'Starting to convert Test Go images \n'); 

Df = cell(1,nClasses);
for d=0:nClasses-1,
  Df{d+1} = fopen(['test' num2str(d) '.ascii'],'w');
end;
  
rawimages = load(testData,'-ascii');
rawlabels = load(testLbls,'-ascii');
rawimages = rawimages';

for j=1:nTest,
  fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
  fprintf(Df{rawlabels(j)+1},'\n');
end;

fprintf(1,'\n');
for d=0:nClasses-1,
  fclose(Df{d+1});
  D = load(['test' num2str(d) '.ascii'],'-ascii');
  fprintf('%5d Digits of class %d\n',size(D,1),d);
  save(['test' num2str(d) '.mat'],'D','-mat');
end;


% Work with training files second  

fprintf(1,'Starting to convert Training Go data \n'); 

Df = cell(1,nClasses);
for d=0:nClasses-1,
  Df{d+1} = fopen(['digit' num2str(d) '.ascii'],'w');
end;

rawimages = load(trainData,'-ascii');
rawlabels = load(trainLbls,'-ascii');
rawimages = rawimages';

  for j=1:nTrain,
    fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
    fprintf(Df{rawlabels(j)+1},'\n');
  end;

fprintf(1,'\n');
for d=0:nClasses-1,
  fclose(Df{d+1});
  D = load(['digit' num2str(d) '.ascii'],'-ascii');
  fprintf('%5d Digits of class %d\n',size(D,1),d);
  save(['digit' num2str(d) '.mat'],'D','-mat');
end;

%dos('rm *.ascii');
