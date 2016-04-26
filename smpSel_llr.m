function [ smpList ] = smpSel_llr( X,nSel,param )
%Representative sample selection using Locally Linear Reconstruction (LLR)
%	 Compared with TED, LLR select samples from X to reconstruct X locally.
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
%	ref: L. Zhang, C. Chen, J. Bu, D. Cai, X. He, and T. S. Huang, "Active
% Learning Based on Locally Linear Reconstruction," IEEE Trans. Pattern
% Anal. Mach. Intell., vol. 33, no. 10, pp. 2026-2038, Oct. 2011.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

mu = 10^(-1); % see Eq. 13 of the ref
nNeighbor = 5; % used in getLLEWeight

defParam

[nSmp,nFt] = size(X);
if nSel >= nSmp, 
	nSel = nSmp;
end

W = getLLEWeight(X,X,nNeighbor);
I = eye(nSmp);
M = I-W;
M = M'*M;
smpList = [];
H = inv(mu*(M+1e-5*I));
A = M*(X*X')*M;

for selIdx = length(smpList)+1:nSel
	z = inf(nSmp,1);
	for p = 1:nSmp
		if ~any(p==smpList)
			tmp = H(p,:)*H(:,p)/2/(1+H(p,p))*I-H;
			z(p) = 1/(1+H(p,p))*H(p,:)*(A*tmp)*H(:,p);
		end
	end
	[val,trIdx1] = min(z);
	smpList = [smpList,trIdx1];
	H = H-H(:,trIdx1)*H(trIdx1,:)/(1+H(trIdx1,trIdx1));
end

end

function W = getLLEWeight(X,Y,nNeighbor)
% ref to Nonlinear Dimensionality Reduction by Locally Linear Embedding,
% Science, 2000
% use the neighbors of Y(i,:) in X to reconstruct it
% use Lagrange multiplier to solve min ||Y(p,:)-W(p,:)*X|| s.t. sum(w(p,:))=1

% X = bsxfun(@rdivide,X,sqrt(sum(X.^2,2)));
% Y = bsxfun(@rdivide,Y,sqrt(sum(Y.^2,2)));
[nSmpX,nFt] = size(X);
nSmpY = size(Y,1);
W = zeros(nSmpY,nSmpX);
if nNeighbor > nSmpX-1, nNeighbor = nSmpX-1; end
for p = 1:nSmpY
	idx = knnsearch(X,Y(p,:),'K',nNeighbor+1);
	if norm(X(idx(1),:)-Y(p,:))<1e-5 % if X contains Y(p,:)
		idx = idx(2:1+nNeighbor);
	else
		idx = idx(1:nNeighbor);
	end
	X1 = X(idx,:);
	A = [2*X1*X1',ones(nNeighbor,1); ones(1,nNeighbor),0];
	b = [2*X1*Y(p,:)';1];
	sol = A\b;
	W(p,idx) = sol(1:end-1);
end

end
