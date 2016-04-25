function [Xnew] = ftProc_zscore_te(model,X)
%Normalize X with the pre-computed mu and std.
%	MODEL: result of FTPROC_ZSCORE_TR.
%	X: a matrix, each row is a sample.
% Return:
% 	XNEW: a matrix, each row is a new feature vector.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

Xnew = bsxfun(@minus,X,model.mu);
Xnew(:,model.validStd) = bsxfun(@rdivide,...
	Xnew(:,model.validStd),model.std(model.validStd));

end

