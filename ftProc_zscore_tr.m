function [Xnew, model] = ftProc_zscore_tr(X,Y,param)
%Normalize X with zscore.
%	Makes each column (feature)of X to be centered to have mean 0 and 
% scaled to have standard deviation 1.
%	X: a matrix, each row is a sample.
%	Y: useless.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
% 	XNEW: a matrix, each row is a new feature vector.
%	MODEL: struct containing coefficients.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

% the minimum std value allowed to normalize a feature
% avoid too small std in case too large normalized values is got
minStd = 0;

defParam

% [Xnew,model.mu,model.std] = zscore(X);
model.mu = mean(X,1);
model.std = std(X,0,1);
model.validStd = (model.std>minStd); 
Xnew = bsxfun(@minus,X,model.mu);
Xnew(:,model.validStd) = bsxfun(@rdivide,Xnew(:,model.validStd),model.std(model.validStd));

end

