nClasses = 361;
sz = 19;
% makebatches;

load presence.mat

aX = reshape(permute(batchdata,[1 3 2]),[],361);
ay = reshape(permute(batchtargets,[1 3 2]),[],361);

[ni,nd] = size(aX);

X = aX(1:ni*0.8,:);
vX = aX(ni*0.8:ni*0.9,:);
tX = aX(ni*0.9:ni,:);

y = ay(1:ni*0.8,:);
vy = ay(ni*0.8:ni*0.9,:);
ty = ay(ni*0.9:ni,:);

clear aX ay

maxepoch = 10;
numhid   = 200; 
restart  = 1;

rbm;


M = vishid;

%[Wl,acclin] = linreg(X,y,vX,vy);
%[Wr,accrbm] = rbmlinreg(X,y,M,vX,vy);

[W,accrig] = ridgeRegression(X,y,1,vX,vy);

accprior = highestPrior(y,vy);