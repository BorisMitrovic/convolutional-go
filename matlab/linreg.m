function [ W,lin_accuracy ] = linreg( X, y, tX,ty )
%LINREG Summary of this function goes here
%   Detailed explanation goes here
[ni,nd] = size(y);

W = X\y;


%eval
pred = tX * W;

[~,best] = max(pred');
[~,true] = max(ty');

lin_accuracy=sum(best==true)/length(true)

end

