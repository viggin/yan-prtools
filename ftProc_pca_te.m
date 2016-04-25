function [Xnew] = ftProc_pca_te(model,X)
%Extract the PCA feature of X based on the trained MODEL.
%	MODEL: result of FTPROC_PCA_TR.
%	X: a matrix, each row is a sample.
% Return:
% 	XNEW: a matrix, each row is a new feature vector.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

Xnew = [];
if ~isempty(X)
	Xnew = bsxfun(@minus,X,model.mu);
	Xnew = Xnew * model.W_prj;
end

end