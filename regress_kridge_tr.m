function [model] = regress_kridge_tr(X,Y,param)
%Kernel ridge regression.
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

lambda = 1; % regularization parameter
ker = 'lin'; % or rbf etc.
sigma = 2^-7; % param for kernel

defParam

[nSmp,nFt] = size(X);
nm = @(X,p)repmat(sum(X.^2,2),1,p);
linKer = @(X1,X2)X1*X2';
rbfKer = @(X1,X2)exp(-sigma*(nm(X1,size(X2,1))+nm(X2,size(X1,1))'-2*X1*X2'));
eval(['kerFun = ' ker 'Ker;'])

b0 = mean(Y);
Yz = Y-b0; % shift Y
[Xz, modelz] = ftProc_zscore_tr(X);
Xz = [ones(nSmp,1),Xz];
KXz = kerFun(Xz,Xz);
R = eye(nSmp)*lambda; % Tikhonov Regularization Matrix
b = (KXz + R) \ Yz;

model.kerFun = kerFun;
model.trXz = Xz;
model.b0 = b0;
model.modelz = modelz;
model.b = b;

end