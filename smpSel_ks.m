function [ smpList ] = smpSel_ks( X,nSel,param )
%Representative sample selection using the Kennard-Stone algorithm
% KS select samples from X that are farthest from each other
% 
%	X : each row is a sample.
%	NSEL : number of samples to select
%	PARAM: struct of parameters, not used here.
% Return:
%	SMPLIST: indices (of rows in X) of the selected samples.
% 
%	ref: R. W. Kennard, and L. A. Stone, "Computer aided design of
% experiments," Technometrics, vol. 11, no. 1, pp. 137-148, Feb. 1969.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

[nSmp,nFt] = size(X);
if nSel >= nSmp, 
	nSel = nSmp;
end

D = squareform(pdist(X));
smpList = [];
[smpList(1),smpList(2)] = find(D==max(max(D)),1,'first');

while length(smpList) < nSel
	minDist = min(D(smpList,:),[],1);
	[maxMinDist,trIdx1] = max(minDist);
	smpList = [smpList,trIdx1];
end

end
