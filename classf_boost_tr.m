function [model] = classf_boost_tr(X,Y,param)
%AdaBoost with stump weak learner.
%	X : each row is a sample.
%	Y : a column vector, class labels for X starting from 1. Only supports
% binary class problems.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	MODEL: a struct containing coefficients.
%  
%	A self-implemented boost classifier using Matlab's object oriented 
% interface. Currently we only have the stump weak learner. Users are
% welcome to implement their own learner according to the interface in
% stump.m
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

maxIter = 50;

defParam

if any(Y ~= 1 & Y ~= 2)
	error('Only supports binary class problems.')
end

Y = (Y-1.5)*2;
model = adaboost(@stump,maxIter);
model.train(X,Y);

end