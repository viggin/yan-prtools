function index_new = incremental_randsamp(n, m, index_old)
% INCREMENTAL_RANDSAMP(N,M) -- return the index of randomly sampled m elements from an n-length array
%                  in which the indices indicated by index_old must not be
%                  part of it


if n < m + length(index_old)
    error('n must not be less than m + length(index_old)!');
end

index_compl = complement_index(n, index_old);   % find the indices of remaining entries
size_avaliab = length(index_compl);             % the size of remaining

index_new = index_compl(randsamp(size_avaliab, m));

if size(index_new, 2) > size(index_new, 1)
    index_new = index_new';
end
