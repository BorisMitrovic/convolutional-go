function [ W ] = linearRegression( X,Y )
%LINEARREGRESSION Summary of this function goes here
%   Detailed explanation goes here

W = (X'*X)\X'*Y; % least squares

end

