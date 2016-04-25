function [pred, prob] = classf_gauss_te(model,Xtest)
%Classify XTEST based on the trained MODEL
%	MODEL: result of CLASSF_GAUSS_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	PRED : predicted labels for XTEST
%	PROB : the confidence (probability) of PRED
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

s = std(model.X);
if all(s>0)
	[pred,trErr,prob] = classify(Xtest, model.X, model.Y, model.type);
else
	error('"classify" requires that the variances in each feature of X must be positive.')
end

end
