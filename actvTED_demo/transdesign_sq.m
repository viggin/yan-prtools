function index = transdesign_sq(K, m, lambda, candidate_index)

% Usage: [index, variance] = transdesign_sq(K, m, candidate_index, lambda)
%
% Sequential transductive experiment design: active learning for kernel
% ridge regression. If kernel is linear, then the algorithm becomes an active
% learning approach for linear ridge regression. 
%
% Inputs:     
%             K--- Kernel matrix of all the data samples
%             m--- Number of data samples to select
%             candidate_index --- IDs of candidate samples from all the samples. 
%                        The default setting is the whole set. By specifing a
%                        random subset, one can save computationl cost.
%             lambda --- A regularization parameter for the kernel ridge regression
% 
% Outputs:
%             index--- Indices of selected data samples
%	06_ICML_Active Learning via Transductive Experimental Design


error( nargchk(1, 4, nargin));
if nargin < 4
    candidate_index = [];
end
if nargin < 3
    lambda = [];
end
if nargin < 2
    m = [];
end
if isempty(m)
    m = 1;
end
if isempty(candidate_index)
    candidate_index = [1:size(K,1)]';
end
if isempty(lambda)
    lambda = 10e-4;
end

n = size(K, 1);
m = min(n, m);
q = length(candidate_index);

index = zeros(m, 1);
fprintf('selecting samples ... ');
for i = 1 : m
    score = zeros(1, q);
    for j = 1 : q
        k = candidate_index(j);
        score(j) = K(k,:)*K(:,k)/(K(k,k) + lambda);
    end
    [dummy, I] = max(score);
    index(i) = candidate_index(I);
    % updata K
    K = K - K(:,index(i))*K(index(i),:)/(K(index(i),index(i)) + lambda);
end
fprintf('done \n');
index = index(1:m);
