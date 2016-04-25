function [pred, prob] = classf_elm_te(model,Xtest)
%Classify XTEST based on the trained MODEL
%	MODEL: result of CLASSF_ELM_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	PRED : predicted labels for XTEST
%	PROB : the output score of PRED.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

nTest = size(Xtest,1);

Xtest = Xtest';
tempH_test = model.InputWeight*Xtest;
tempH_test = tempH_test + repmat(model.BiasofHiddenNeurons,1,nTest);
H_test = model.actFun(tempH_test);

prob = H_test' * model.OutputWeight;
[maxProb,pred] = max(prob,[],2);

end