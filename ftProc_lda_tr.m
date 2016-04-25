function [Xnew, model] = ftProc_lda_tr(X,Y,param)
%Training Linear Discriminant Analysis (LDA).
%	X: a matrix, each row is a sample.
%	Y: a column vector, class label for X starting from 1.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
% 	XNEW: a matrix, each row is a new feature vector.
%	MODEL: struct containing coefficients.
% 
%	Use the generalized eigenvalue algorithm in Matlab eig(Sb,Sw). Although
% it can handle singular Sw, but sometimes do PCA before LDA will help.
% The function do not use PCA before LDA, only subtracts the mean.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

nClass = length(unique(Y));
postLdaDim = nClass-1;  %the dimension of projected subspace.

defParam

[nSmp,nFt] = size(X);
postLdaDim = min(postLdaDim,nFt); % could not exceed nFt

mu = mean(X,1);
X = bsxfun(@minus,X,mu);

% sort X by label
[Y, I] = sort(Y);
X = X(I,:);
[~, I ,~] = unique(Y,'legacy');

nSmpCls = [I(1);diff(I)];
J = I-nSmpCls+1;

% calculate within-class average
delta_X = X;
mu_per_class = zeros(nClass,nFt);
for p = 1:nClass
	mu_per_class(p,:) = mean(delta_X(J(p):I(p),:),1);
	delta_X(J(p):I(p),:) = delta_X(J(p):I(p),:)-...
		repmat(mu_per_class(p,:),nSmpCls(p),1);
end

% within class scatter matrix
Sw = delta_X'*delta_X;

% between class scatter matrix
Sb = zeros(nFt);
for p = 1:nClass
	Sb = Sb+nSmpCls(p)*mu_per_class(p,:)'*mu_per_class(p,:); 
end

%  Do LDA,Sb*V = lambda*Sw*V
[LDA_V, LDA_D] = eig(Sb,Sw);
LDA_D = diag(LDA_D);
LDA_D(isnan(LDA_D)) = -inf;
[LDA_D, I] = sort(LDA_D,'descend');
LDA_V = LDA_V(:,I);

model.W_prj = LDA_V(:,1:postLdaDim);
model.mu = mu;
model.eigVals = LDA_D';
model.postLdaDim = postLdaDim;
model.param = param;

Xnew = X*model.W_prj;

end