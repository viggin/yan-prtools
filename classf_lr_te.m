function [pred, prob] = classf_lr_te(model,Xtest)
%Classify XTEST based on the trained MODEL
%	MODEL: result of CLASSF_LR_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	PRED : predicted labels for XTEST
%	PROB : the confidence (probability) of PRED
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

Xtest = [ones(size(Xtest,1),1),Xtest];
prob = sigmoid(Xtest*model.thetas);
if size(model.thetas,2) == 1
	pred = prob >= .5;
	pred = 2-pred;
	prob = [prob,1-prob];
else
	[maxProb,pred] = max(prob,[],2);
end

end
