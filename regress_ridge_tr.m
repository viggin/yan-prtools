function [model] = regress_ridge_tr(X,Y,param)
%Ridge regression. 
% Can set separate regularization for each ft according to prior knowledge.
% Can also set weight for each sample.
% 
%	X : each row is a sample.
%	Y : a column vector.
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
smpWt = ones(nSmp,1); % importance of each sample. The larger the more important
lambda = 1; % regularization parameter

defParam

W = diag(smpWt.^2);

b0 = mean(Y);
Yz = Y-b0; % shift Y
[Xz, muZ, sigmaZ] = zscore(X); % scale X
R = diag(ftPenal.^2)*lambda; % Tikhonov Regularization Matrix
b = (Xz'*W*Xz + R) \ (Xz' *W* Yz);

% scale and shift back
bs = b;
bs(sigmaZ~=0) = b(sigmaZ~=0)./sigmaZ(sigmaZ~=0)';
model.b = [b0-muZ*bs; bs];

end