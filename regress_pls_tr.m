function [model] = regress_pls_tr(X,Y,param)
%Wrapper of patial least square regression in matlab. 
% Popular in chemometrics. Suitable when variables are largely correlated.
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

numComp = 5; % number of PLS components

defParam

[XL,yl,XS,YS,beta,PCTVAR] = plsregress(X,Y,numComp);
% plot(1:10,cumsum(100*PCTVAR(2,:)),'-bo');
% xlabel('Number of PLS components');
% ylabel('Percent Variance Explained in y');

model.beta = beta;

end
