function [ftRank,ftScore] = ftSel_single(ft,target,param)
%Feature ranking based on each single feature's prediction accuracy
%	FT : sample matrix, each row is a sample.
%	TARGET : a column vector, depend on the wrapper prediction algorithm
% used.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	FTRANK: rank of the features (most important first)
%	FTSCORE: the accuracy of features corresponding to FTRANK.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

nCv = 5;

defParam

cvObj = cvpartition(target,'k',nCv);
userdata.cvObj = cvObj;
userdata.ft = ft;
userdata.target = target;

nFt = size(ft,2);
errs = zeros(1,nFt);
for p = 1:nFt
	x = false(1,nFt);
	x(p) = true;
	errs(p) = getErrRate_example(x,userdata);
end

[ftScore,ftRank] = sort(errs,'ascend');

end