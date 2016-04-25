function [model] = classf_knn_tr(X,Y,param)
%Train a K-nearest-neighbor classfier.
%	X : each row is a sample.
%	Y : a column vector, class labels for X starting from 1.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	MODEL: a struct containing coefficients.
% 
%	Note: KNN classifier is a 'lazy' classifier. It actually needs no
%	training, so we simply save all the training samples in MODEL.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

K = 1; % how many nearest neighbors. Should be odd.
defParam

model.K = K;
model.nClass = max(Y);
model.X = X;
model.Y = Y;

end