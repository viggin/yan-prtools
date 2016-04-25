function B = submatrix(A, index1, index2)

% USAGE: B = submatrix(A, index1, index2)
% pick out a submatrix B from A

n = length(index1);
m = length(index2);
B = zeros(n, m);

for i = 1 : n
    B(i, :) = A(index1(i), index2);
end
    
    
    