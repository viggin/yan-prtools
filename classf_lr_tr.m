function [model] = classf_lr_tr(X,Y,param)
%Train a logistic regression classfier
% 
%	Weighted regularized logistic regression, one vs. all for multi-class
% problems (this strategy is often better than softmax).
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
% the feature will be less relied in the model. If you do not know the
% importance of features in prior, just set to all ones.
lambda = 0; % regularization parameter
nIter = 100; % number of iteration. Should be larger if nSmp or nFt is large

defParam

X = [ones(nSmp,1),X]; % add constant column
nCls = max(Y);
initTheta = zeros(nFt+1,1);
options = optimset('GradObj','on','MaxIter',nIter);
fmin = @fmincg; % fminunc is very slow

% similar to fmincg
% options = struct('GradObj','on','MaxIter',param.nIter,'Display','off',...
% 	'Method','cg','DerivativeCheck','off');
% fmin = @minFunc;
% warning off

if nCls == 2 % only compute one theta, will be faster
	f = @(t)lrCostFunction(t,X,(Y==1),lambda*ftPenal); % the 3rd para should be 0 or 1
	[thetas,fh] = fmin(f,initTheta,options);
else
	thetas = zeros(nFt+1,nCls); % one vs all
	for p = 1:nCls
		f = @(t)lrCostFunction(t,X,(Y==p),lambda*ftPenal); % the 3rd para should be 0 or 1
		thetas(:,p) = fmin(f,initTheta,options);
	end
end

model.thetas = thetas;

end

function [J, grad] = lrCostFunction(theta, X, Y, reguTerm)

smpNum = length(Y);
hyp = sigmoid(X*theta); % hypothesis
penalTerm = sum(theta(2:end).^2 .* reguTerm')/2/smpNum;
J = -sum(Y.*log(hyp)+(1-Y).*log(1-hyp)) / smpNum + penalTerm;
penalTermG = theta.*[0;reguTerm']/smpNum; % penalty for theta(1) is 0
grad = X'*(hyp-Y)/smpNum + penalTermG;

end
