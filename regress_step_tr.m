function [model] = regress_step_tr(X,Y,param)
%Wrapper of stepwisefit in matlab.
% Please search the doc of stepwisefit for more configurations.
% 
%	X : each row is a sample.
%	Y : a column vector.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	MODEL: a struct containing coefficients.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

[model.b,model.se,model.pval,model.inmodel,model.stats] = ...
	stepwisefit(X,Y,'display','off');
% inmodel: features selected

end
