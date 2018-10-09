function [accuracy_prior] = highestPrior(y,ty)

[~,i]=max(y');
best = mode(i);

[~,true]=max(ty');

accuracy_prior = sum(best==true)/length(true)

end