function [ftRank,ftScore] = ftSel_mrmr(ft,label,param)
%Feature ranking using minimum redundancy maximal relevance (mRMR).
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
%	mRMR and mutual information toolbox by Hanchuan Peng
%	http://www.mathworks.com/matlabcentral/fileexchange/?term=authorid%3A27911
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

[nSmp,nFt] = size(ft);
nFtSel = min(50,nFt); % number of features to select. If too many, could be slow.
method = 'miq'; % two variants of mRMR

defParam

% discretize data
m = mean(ft,1);
m = repmat(m,nSmp,1);
d = std(ft,0,1);
d = repmat(d,nSmp,1);
% ft_d = -double(ft<m-d)-(ft<m)*1+(ft>=m)+(ft>=m+d); % 4 level alternative
ft_d = -double(ft<m-d)+(ft>=m+d); % seems a little better than 4 level

addpath(genpath('mRMR'))

if strcmp(method,'mid')
	[ftRank] = mrmr_mid_d(ft_d,label,nFtSel);
else
	[ftRank] = mrmr_miq_d(ft_d,label,nFtSel);
end
ftScore =[];

end