function [ftRank,ftScore] = ftSel_fisher(ft,label,param)
%Feature ranking using Fisher ratio.
%	
%	FT : sample matrix, each row is a sample.
%	LABEL : a column vector, only for binary classification. LABEL must be 1 or 2.
%	PARAM: struct of parameters, useless here.
% Return:
%	FTRANK: rank of the features (most important first)
%	FTSCORE: the estimated scores of the features corresponding to FTRANK.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

I1 = (label==1);
I2 = (label==2);

m1 = mean(ft(I1,:),1);
m2 = mean(ft(I2,:),1);

s1 = std(ft(I1,:),0,1);
s2 = std(ft(I2,:),0,1);

J = (m1-m2).^2./(s1.^2+s2.^2);
J(isnan(J)) = -1;

[ftScore, ftRank] = sort(J,'descend');

end