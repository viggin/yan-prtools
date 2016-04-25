function [model] = classf_softmax_tr(X,Y,param)
%Softmax classifier.
%	Weighted regularized softmax, extension to logistic regression
% 
%	X : each row is a sample.
%	Y : a column vector, class labels for X starting from 1.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	MODEL: a struct containing coefficients.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

[nSmp,nFt] = size(X);

ftPenal = ones(1,nFt); % penalization weight of each feature. The larger,
% the feature will be less relied in the model
lambda = 0; % regularization parameter
nIter = 500; % number of iteration. Should be larger if nSmp or nFt is large

defParam

X = [ones(nSmp,1),X]; % add constant column
nCls = max(Y);

initTheta = zeros((nFt+1)*nCls,1);
options = optimset('GradObj','on','MaxIter',nIter);
fmin = @fmincg; % fminunc is very slow

% % similar to fmincg
% % minFunc toolbox: https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html
% options = struct('GradObj','on','MaxIter',nIter,'Display','off',...
% 	'Method','lbfgs','DerivativeCheck','off');
% fmin = @minFunc;

Ym = full(ind2vec(Y')');
f = @(t)lrCostFunction(t,X,Ym,lambda*ftPenal); % the 3rd para should be 1~K
[thetas,fh] = fmin(f,initTheta,options);

model.thetas = reshape(thetas,nFt+1,nCls);

end

function [J, grad] = lrCostFunction(theta0, X, Ym, lambda)

nFt1 = size(X,2);
[smpNum,classNum] = size(Ym);
theta = reshape(theta0,nFt1,classNum);
hyp = hypothesis(X*theta); % hypothesis
penalTerm = sum(lambda*(theta(2:end,:).^2))/2;
cost = -sum(sum(Ym.*log(hyp))) / smpNum;
J = cost + penalTerm;

penalTermG = theta0.*repmat([0;lambda'],classNum,1); % penalty for theta(1) is 0
costG = X'*(hyp-Ym)/smpNum;
grad = costG(:) + penalTermG;

end

function g = hypothesis(z)

g = exp(z);
g = bsxfun(@rdivide,g,sum(g,2));

end
