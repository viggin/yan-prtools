function [ smpList ] = smpSel_cluster( X,nSel,param )
%Representative sample selection based on cluster centers
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
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

method = 1; % 1: Agglomerative cluster, 2: kmeans cluster
defParam

[nSmp,nFt] = size(X);
if nSel >= nSmp, 
	nSel = nSmp;
end

if method == 1
	IDX = clusterdata(X,'maxclust',nSel,'linkage','average'); % Agglomerative cluster
elseif method == 2
	[IDX,C] = kmeans(X,nSel); % kmeans cluster
end

for p = 1:nSel
	idx1 = find(IDX==p);
	idx2 = selCenterSmp(X(idx1,:));
	smpList(p) = idx1(idx2);
end

end

function idx = selCenterSmp(data)

if size(data,1) == 1, idx = 1;
else
	med = median(data,1);
	idx = knnsearch(data,med);
end

end