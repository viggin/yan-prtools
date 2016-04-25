function [pred, prob] = classf_ann_te(model,Xtest)
%Classify XTEST based on the trained MODEL
%	MODEL: result of CLASSF_ANN_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	PRED : predicted labels for XTEST
%	PROB : the output score of PRED
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

% prob = sim(model,Xtest);
prob = model(Xtest')';
[maxProb,pred] = max(prob,[],2);

end
