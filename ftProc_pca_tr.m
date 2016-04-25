function [Xnew, model] = ftProc_pca_tr(X,Y,param)
%Training Principal Component Analysis (PCA).
%	X: a matrix, each row is a sample.
%	Y: useless.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
% 	XNEW: a matrix, each row is a new feature vector.
%	MODEL: struct containing coefficients.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com


% if 0, extract all principal components;
% elseif integer, extract pcaCoef PCs;
% elseif >0 and <1, extract PC with energy ratio of pcaCoef.
pcaCoef = 0;

defParam

[nSmp,nFt] = size(X);
mu = mean(X,1);
% for p = 1:train_num % dealing with out of memory error
% 	ft(:,p) = ft(:,p)-mu_total0;
% end
X = bsxfun(@minus,X,mu);

% select the faster way to do PCA
if nFt > nSmp
	[V1, D1] = eig(X*X'/(nFt-1));
else
	[V1, D1] = eig(X'*X/(nFt-1));
end
D1 = diag(D1);
[evs,I] = sort(D1,'descend');
V1 = V1(:,I);

D1 = cumsum(evs);
D1 = D1/D1(end);
% calculate the dimension of the data after PCA projection
if pcaCoef == 0
	postPcaDim = min(nSmp,nFt);
elseif pcaCoef > 0 && pcaCoef < 1
	pcaRate = pcaCoef;
	postPcaDim = nnz(D1 < pcaRate)+1;
elseif pcaCoef >= 1 && floor(pcaCoef) == pcaCoef
	postPcaDim = pcaCoef;
end
	
V1 = V1(:,1:postPcaDim);
if nFt > nSmp
	W_prj = X'*V1; % W_pca: nFt-by-postPcaDim
else
	W_prj = V1;
end
W_prj = bsxfun(@rdivide,W_prj,sqrt(sum(W_prj.^2,1)));
clear V1

% Enforce a sign convention on the coefficients - the largest element in each
% column will have a positive sign. -- borrowed from princomp
[~,maxind] = max(abs(W_prj),[],1);
colsign = sign(W_prj(maxind + (0:nFt:(postPcaDim-1)*nFt)));
W_prj = W_prj.*repmat(colsign,nFt,1);

% project X to PCA subspace
Xnew = X*W_prj;

model.mu = mu;
model.W_prj = W_prj;
model.eigVals = evs;
model.postPcaDim = postPcaDim;
model.param = param;

end
