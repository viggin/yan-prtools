function [model] = regress_rf_tr(X,Y,param)
%Random forest. 
% 
%	X : each row is a sample.
%	Y : a column vector.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	MODEL: a struct containing coefficients.
%	Dependence: RandomForest toolbox by Abhishek Jaiantilal
% https://code.google.com/p/randomforest-matlab/
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

nTrees = 100;
mtry = floor(sqrt(size(X,2))); % number of predictors sampled for spliting at each node.

defParam

%% matlab built-in. slow.
% b = TreeBagger(numTrees,X,Y,'Method','regression',...
% 	'OOBVarImp','on');
% plot(oobError(b));
% regrs_value = predict(b,Xtest);

%% Andy Liaw et al.'s
addpath RandomForest-v0.02\RF_Reg_C
opt.do_trace = 1;
model = regRF_train(X,Y,nTrees,mtry,opt);

end
