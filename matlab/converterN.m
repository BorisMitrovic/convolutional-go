% Version 1.1
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

name = 'single_';%'presence_';
data = '.txt';
lbls = '.lbl';

nClasses = 361;
sz = 19;
nSplits = 1;


% Work with test files first 
fprintf(1,'Go data \n'); 

for tr=0:1,
set = '';
if tr,
  set = 'train';
else
  set = 'test';
end    
    
fprintf(1,['Starting to convert ', set, ' Go images \n']); 

Df = cell(1,nClasses);
for d=0:nClasses-1,
  Df{d+1} = fopen([set num2str(d) '.ascii'],'w');
end;

for i=0:nSplits-1,
  fprintf(1,'.');
  tData = [name,set,num2str(i),data];
  tLbls = [name,set,num2str(i),lbls];
    
  rawimages = load(tData,'-ascii');
  rawlabels = load(tLbls,'-ascii');
  rawimages = rawimages';
  nSet = length(rawlabels);

  for j=1:nSet,
    fprintf(Df{rawlabels(j)+1},'%3d ',rawimages(:,j));
    fprintf(Df{rawlabels(j)+1},'\n');
  end;
end

fprintf(1,'\n');
for d=0:nClasses-1,
  fclose(Df{d+1});
  D = load([set num2str(d) '.ascii'],'-ascii');
  fprintf('%5d Digits of class %d\n',size(D,1),d);
  save([set num2str(d) '.mat'],'D','-mat');
  dos(['rm ' set num2str(d) '.ascii']);
end;

end
