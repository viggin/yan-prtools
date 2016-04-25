function [model] = regress_lasso_tr(X,Y,param)
%LASSO regression in matlab. L1 regularization. 
% Can be used to select useful features. but sometimes not stable.
% 
%	X : each row is a sample.
%	Y : a column vector.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	MODEL: a struct containing coefficients.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

lambda = []; % if set to [], lasso will try a series of lambda

defParam

[B,FitInfo] = lasso(X,Y,'Lambda',lambda);
cIdx = find(FitInfo.MSE==min(FitInfo.MSE),1,'last');

model.B = B(:,cIdx);
model.Intercept = FitInfo.Intercept(cIdx);

end
