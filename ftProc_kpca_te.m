function [Xnew] = ftProc_kpca_te(model,X)
%Extract the KPCA feature of X based on the trained MODEL.
%See FTPROC_KPCA_TR.
%	MODEL: result of FTPROC_KPCA_TR.
%	X: a matrix, each row is a sample.
% Return:
% 	XNEW: a matrix, each row is a new feature vector.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

Xnew = [];
if ~isempty(X)
	K = model.kerFun(X,model.Xtrain);
	Xnew = K * model.W_prj;
end

end