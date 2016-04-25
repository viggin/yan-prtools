function [index, rest] = randsamp(n, m)
% USAGE: 
% [index, rest] = RANDSAMP(N,M)  
% return the index of randomly sampled m elements from an n-length array
%

if n < m
    error('n must not be less than m!');
end

index = randperm(n);
index = index(1:m);
rest = complement_index(n, index);