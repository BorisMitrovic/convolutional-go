function [ row ] = getTarget( i, nClasses, nPerClass )
%GETTARGET Summary of this function goes here
%   Detailed explanation goes here

  row = zeros(1,nClasses); row(i) = 1;  

end

