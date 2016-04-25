function [score, roc_x, roc_y] = roc_curve(Ypred, Ytruth, k)

% compute the ROC curve and score
%
% Inputs:   
%            Ypred --- a column real-value vector
%            Ytruth --- a column vector, the ground truth, each entry is -1 or +1
%
% Outputs:
%            score --- a scalar, 
%            roc_x --- sensitivity
%            roc_y --- 1-specificity

N = length(Ypred);
if length(Ytruth) ~= N
    error('The length of two input vectors shoule be the same!');
end
if k-1 >= N
    error('k-1 must be less than the length of Ypred');
end

pos_num = sum(Ytruth == 1);
neg_num = sum(Ytruth == -1);

if pos_num*neg_num == 0
    error('There must be at least one +1 or -1 in Ytruth!');
end

interval = round(N/(k-1));
top_n = [0:interval:N];

roc_x = zeros(length(top_n),1);
roc_y = zeros(length(top_n),1);

[dummy, order] = sort(-Ypred);
for t = 1 : length(top_n)
    % test the precision in varying top_n window size
    hits = sum(Ytruth(order(1:top_n(t))) == 1);
    wronghits = sum(Ytruth(order(1:top_n(t))) == -1);
    % precision(uid, t) = hits/top_n(t);                   % precision
    roc_y(t) = hits / pos_num;                             % sentitivity
    roc_x(t) = wronghits / neg_num;                        % 1-specificity
end

score = aucroc(roc_x, roc_y, 100);