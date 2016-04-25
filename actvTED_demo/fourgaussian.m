function X = fourgaussian()
%
% generate a toy problem with 4 gaussians

rand('seed', 0);
randn('seed', 0);

D1 = randn(50,2);
D2 = 1.5*randn(50,2) + repmat([5,5], 50, 1);
D3 = 1.2*randn(50,2) + repmat([-2.5,7.5], 50, 1);
D4 = 1.4*randn(50,2) + repmat([5,-1.5], 50, 1);
X = [D1; D2; D3; D4];
