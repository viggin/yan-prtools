function index_B = complement_index(totalsize, index_A)

% Usage: index_B = complement_index(totalsize, index_A)
% return the complementary indices given exisiting indices A

dummy = ones(1, totalsize);
dummy(index_A) = 0;
index_B = find(dummy);