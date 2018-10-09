function [ W, ridge_accuracy ] = ridgeRegression( X,Y,L,tX,ty )
%LINEARREGRESSION Summary of this function goes here
%   Detailed explanation goes here
I  = eye(size(X,361));

W = (X'*X + L*I)\X'*Y ; % least squares

pred = tX * W;

[~,best] = max(pred');
[~,true] = max(ty');

ridge_accuracy = sum(best==true)/length(true)

end

