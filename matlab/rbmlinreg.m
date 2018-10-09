function [ W,rbm_accuracy ] = rbmlinreg( X, y, M, tX,ty )
%LINREG Summary of this function goes here
%   Detailed explanation goes here
[ni,nd] = size(y);

Xt = X*M;

W = Xt\y;

%eval

pred = (tX*M) * W;

[~,best] = max(pred');
[~,true] = max(ty');

rbm_accuracy=sum(best==true)/length(true)

end

