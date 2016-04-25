function [ftRank,ftScore] = ftSel_corr(ft,target,param)
%Feature ranking based on correlation coefficients.
%	FT : sample matrix, each row is a sample.
%	TARGET : a column vector, can be continuous for regression or 1/2 for 
% binary classification.
%	PARAM: struct of parameters, useless here.
% Return:
%	FTRANK: rank of the features (most important first)
%	FTSCORE: the estimated scores of the features corresponding to FTRANK.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

R = corrcoef([ft,target]);
R1 = abs(R(end,1:end-1));
R1(isnan(R1)) = -1;

[ftScore, ftRank] = sort(R1,'descend');

end