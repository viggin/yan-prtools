function [model] = regress_simplefit_tr(X,Y,param)
%Wrapper to matlab's fitting functions
%	Include least squares, robust fitting, quadratic terms, etc. 
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

method = 3;
isRobust = 0;

defParam

if method == 1 % robust fit, univariate or multivariate
	model.coefs = robustfit(X,Y);
elseif method == 2 % univariate polynomial fit
	if size(X,2) > 1, error('only allow univariate'); end
	model.coefs = polyfit(X,Y,1);
elseif method == 3 % general method. See doc for LinearModel.
	if isRobust, v = 'on'; else v = 'off'; end
	model.coefs = LinearModel.fit(X,Y,'RobustOpts',v);
%   model = LinearModel.fit(X,Y,'purequadratic'); % include sqr, lin and constant	
end

model.method = method;

end