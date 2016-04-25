function [pred, prob] = classf_boost_te(model,Xtest)
%Classify XTEST based on the trained MODEL
%	MODEL: result of CLASSF_BOOST_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	PRED : predicted labels for XTEST
%	PROB : the confidence (probability) of PRED
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

pred = model.test(Xtest);
pred = pred/2+1.5;
prob = [];

end
