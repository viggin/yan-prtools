function [ftSubset,ftScore] = ftSel_stepwisefit(ft,target)
%Feature selection based on Matlab's stepwisefit
%	FT : sample matrix, each row is a sample.
%	TARGET : a column vector, should be regression target, binary-class
% labels may also work.
%	PARAM: struct of parameters. not used here.
% Return:
%	FTSUBSET: ordered selected features (most important first)
%	FTSCORE: not returned.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

[b,se,pval,inmodel,stats,nextstep,history] = ...
	stepwisefit(ft,target,'display','off',...
	'penter',.02);
order = sum(history.in,1);
[~,idx] = sort(order,'descend');
n = nnz(order>0);
ftSubset = idx(1:n); % selected in model
ftScore = [];

end