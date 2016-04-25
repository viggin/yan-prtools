function [ smpList ] = smpSel_ted( X,nSel,param )
%Representative sample selection using transductive experimental design (TED)
% TED select minimum samples from X to reconstruct X
% 
%	X : each row is a sample.
%	NSEL : number of samples to select
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	SMPLIST: indices (of rows in X) of the selected samples.
% 
%	Dependence: actvTED_demo, http://www.dbs.ifi.lmu.de/~yu_k/ted/
%	ref: Active Learning via Transductive Experimental Design, ICML, 2006
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

addpath actvTED_demo

useRbf = 0; % linear or rbf kernel
sigma = 1; % in rbf kernel
lambda = 1; % see the ref

% TED actually allows to select samples from a subset of X to reconstruct X
% this can be done by assigning candidate_index. Default candidates are all
% samples in X
[nSmp,nFt] = size(X);
candidate_index = 1:nSmp;

defParam

if nSel > length(candidate_index)
	nSel = length(candidate_index); 
end

data = X;
if ~useRbf
	K = data*data';
else
	K = rbfkernel(data,data,sigma);
end

smpList = transdesign_sq(K,nSel,lambda, candidate_index');

end

