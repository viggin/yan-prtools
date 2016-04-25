function Kxy = rbfkernel(X, Y, sigma)
% RBFKERNEL		Calculate RBF kernel between data matrix X and Y
%
%	DESCRIPTION
%	Calculate RBF kernel w.r.t. each data point in matrix X and Y. The
%	calculatioin is K(x,y) = exp(-1/sigma^2 * norm(x-y)).
%
%	INPUT
%		X: N by K matrix where each column vector is a data point
%		Y: M by K matrix where each column vector is a data point
%		SIGMA: The parameter in the kernel calculation
%
%	OUTPUT
%		K: N by M matrix, each entry K(i,j) is the kernel function on data
%		points X(i,:) and Y(j,:)
%



[N, K] = size(X);
M = size(Y, 1);

Kxy = (ones(M, 1) * sum((X.^2)', 1))' + ...
  ones(N, 1) * sum((Y.^2)',1) - ...
  2.*(X*(Y'));

Kxy = exp(-0.5 * Kxy / sigma^2);