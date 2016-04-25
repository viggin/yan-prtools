function [ftSubset,ftScore] = ftSel_boost(ft,target,param)
%Feature selection using AdaBoost with the stump weak learner
% 
%	FT : sample matrix, each row is a sample.
%	TARGET : a column vector, class labels for X starting from 1. Only supports
% binary class problems.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	FTSUBSET: ordered selected features (most important first)
%	FTSCORE: not returned.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

[nSmp,nFt] = size(ft);
nSel = min(10,nFt); % number of features to select

defParam

if any(target ~= 1 & target ~= 2)
	error('Only supports binary class problems.')
end

Y = (target-1.5)*2;
model = adaboost(@stump,nSel);
model.train(ft,Y);

nSel = min(nSel,model.realLen);
ftSubset = nan(1,nSel);
for p = 1:nSel
	ftSubset(p) = model.weakLearners{p}.dim;
end
ftScore = model.alpha;

end
