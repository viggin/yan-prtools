function [pred, prob] = classf_tree_te(model,Xtest)
%Classify XTEST based on the trained MODEL
%	MODEL: result of CLASSF_LR_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	PRED : predicted labels for XTEST
%	PROB : the score of PRED
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

[pred, prob] = predict(model,Xtest);

end
