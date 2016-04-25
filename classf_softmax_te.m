function [pred,prob] = classf_softmax_te(model,Xtest)
%Classify XTEST based on the trained MODEL
%	MODEL: result of CLASSF_SOFTMAX_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	PRED : predicted labels for XTEST
%	PROB : the confidence (probability) of PRED
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

Xtest = [ones(size(Xtest,1),1),Xtest];
prob = hypothesis(Xtest*model.thetas);
[maxProb,pred] = max(prob,[],2);

end


function g = hypothesis(z)

g = exp(z);
g = bsxfun(@rdivide,g,sum(g,2));

end
