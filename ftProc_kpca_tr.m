function [Xnew, model] = ftProc_kpca_tr(X,Y,param)
%Training Kernal Principal Component Analysis (KPCA).
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
kernel = 'rbf'; % 'lin','poly','rbf','lap'
gamma = 1; % kernel parameters for rbf, lap, poly. See below
degree = 2; % kernel parameters for poly. See below

defParam

nm = @(X,p)repmat(sum(X.^2,2),1,p);
linKer = @(X1,X2)X1*X2';
rbfKer = @(X1,X2)exp(-(nm(X1,size(X2,1))+nm(X2,size(X1,1))'-2*X1*X2')/2/gamma^2);
lapKer = @(X1,X2)exp(-pdist2(X1,X2)/gamma); % Laplacian
polyKer = @(X1,X2)(1+gamma*X1*X2').^degree;
if strcmpi(kernel,'lin'), kerFun = linKer;
elseif strcmpi(kernel,'poly'), kerFun = polyKer;
elseif strcmpi(kernel,'rbf'), kerFun = rbfKer;
elseif strcmpi(kernel,'lap'), kerFun = lapKer; % Laplacian
else error('unknown ker'); end

K = kerFun(X,X);
[nSmp,nFt] = size(X);
H = ones(nSmp)/nSmp;
Kcentered = K-K*H-H*K+H*K*H;
[V,D] = eig(Kcentered);

D = real(diag(D));
[evs,I] = sort(D,'descend');
V = real(V(:,I));

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

% normalize the projection vectors with nonzero eigenvalues
W_prj = V(:,1:postPcaDim);
nzev = evs(1:postPcaDim)>1e-6;
W_prj(:,nzev) = W_prj(:,nzev)./repmat(sqrt(evs(nzev))',nSmp,1);

% project X to KPCA subspace
Xnew = K*W_prj;

model.Xtrain = X;
model.W_prj = W_prj;
model.eigVals = evs;
model.postPcaDim = postPcaDim;
model.kerFun = kerFun;

end
