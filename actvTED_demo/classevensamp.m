function index = classevensamp(Y, k)

% USAGE: index = classevensamp(Y, k)
% Sample k random examples for each class, usually used for initialization
% of an experiment
% 
% Input:
%          Y   --- N x M matrix, N data samples, M classes, each entry
%                  either +1 or 0
%          k   --- size of random samples for each class
% 
% Output:
%          index --- indices of selected data


[N, M] = size(Y);                       % Y's entrie must be +1 or -1 
indicator = zeros(N);
% initialize labeled samples, to ensure at least one class having one
% example
for classid = 1 : M
    posid = find(Y(:,classid) == 1);
    indicator(posid(randsamp(length(posid), min(k, length(posid))))) = 1;
end
index = find(indicator);
    
    