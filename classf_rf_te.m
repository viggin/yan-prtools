function [pred, prob] = classf_rf_te(model,Xtest)
%Classify XTEST based on the trained MODEL
%	MODEL: result of CLASSF_RF_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	PRED : predicted labels for XTEST
%	PROB : votes - unnormalized weights for the model
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

addpath RandomForest-v0.02\RF_Class_C
[pred,prob] = classRF_predict(Xtest,model);

end
