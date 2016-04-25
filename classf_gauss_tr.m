function [model] = classf_gauss_tr(X,Y,param)
%Wrapper of the classify function in matlab.
% 
%	Methods like naive Bayes, fitting normal density function, Mahalanobis 
% distance, etc.
%	X : each row is a sample.
%	Y : a column vector, class labels for X starting from 1.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	MODEL: a struct containing coefficients.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

% algorithm to use, see the doc of classify or the bottom of the code
% type = 'quadratic';
type = 'diagquadratic';
% type = 'linear';
% type = 'mahalanobis';

defParam

model.X = X;
model.Y = Y;
model.type = type;

end

% type:
% linear ！ Fits a multivariate normal density to each group, with a pooled estimate of covariance.
%	This is the default.
% diaglinear ！ Similar to linear, but with a diagonal covariance matrix estimate (naive Bayes classifiers).
% quadratic ！ Fits multivariate normal densities with covariance estimates stratified by group.
% diagquadratic ！ Similar to quadratic, but with a diagonal covariance matrix estimate (naive Bayes classifiers).
% mahalanobis ！ Uses Mahalanobis distances with stratified covariance estimates.