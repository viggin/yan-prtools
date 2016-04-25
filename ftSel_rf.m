function [ftRank,ftScore] = ftSel_rf(ft,target,param)
%Feature ranking using random forest
% 
%	FT : sample matrix, each row is a sample. Will be discretized.
%	LABEL : a column vector, only for classification, must be categorical.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	FTRANK: rank of the features (most important first)
%	FTSCORE: not calculated.
% 
% Dependence:
%	RandomForest toolbox by Abhishek Jaiantilal
% https://code.google.com/p/randomforest-matlab/
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

% classification or regression problem
isRegression = 0;

nTrees = 50;
mtry = floor(sqrt(size(ft,2))); % number of predictors sampled for spliting 
								% at each node
defParam

opt.importance = 1;
opt.do_trace = 0;

if ~isRegression
	addpath RandomForest-v0.02\RF_Class_C
	model = classRF_train(ft,target,nTrees,mtry,opt);
else
	addpath RandomForest-v0.02\RF_Reg_C
	model = regRF_train(ft,target,nTrees,mtry,opt);
end

w = model.importance(:,3);
[ftScore,ftRank] = sort(w,'descend');

end