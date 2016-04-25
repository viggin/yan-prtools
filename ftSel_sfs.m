function [ftSubset,ftScore] = ftSel_sfs(ft,target,param)
%Feature selection using sequential forward selection
%	FT : sample matrix, each row is a sample.
%	TARGET : a column vector, depend on the wrapper prediction algorithm
% used.
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
nCv = 5; % number of cross-validation when evaluating features
nSel = min(10,nFt); % number of features to select
showVerboseInfo = 1;

defParam

cvObj = cvpartition(target,'k',nCv);
userdata.cvObj = cvObj;
userdata.ft = ft;
userdata.target = target;

selFt = [];
perfPerFt = zeros(1,nSel);
perfHistory = zeros(nSel,nFt);
selTimeCost = zeros(1,nSel);

if showVerboseInfo, fprintf('current error: '); end
for sfIdx = 1:nSel
	err1 = inf(1,nFt);
	
	tic
	for ftIdx = 1:nFt
		if any(ftIdx==selFt), continue; end
		curFt = [selFt ftIdx];
		x = false(1,nFt);
		x(curFt) = true;
		err1(ftIdx) = getErrRate_example(x,userdata);
	end
	selTimeCost(sfIdx) = toc;
	
	perfHistory(sfIdx,:) = err1;
	[perfPerFt(sfIdx),selFt(sfIdx)] = min(err1);
	if showVerboseInfo,fprintf('%.4f,',perfPerFt(sfIdx)); end
	
end

fprintf('\n')
ftSubset = selFt;
ftScore = [];

end
