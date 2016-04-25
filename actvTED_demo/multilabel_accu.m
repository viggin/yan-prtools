function Performance = multilabel_accu(Ypred, Ytest, nmiss)
% 
% Usage: Performance = multilabel_accu(Ypred, Ytest, nmiss)
% Evaluate the results of multi-label prediction
%
% Inputs:
%          Ypred --- n x m matrix, predicted values, should be
%                    pre-centerized at 0
%          Ytest --- n x m matrix, desired labels, +1 or -1
%          nmiss --- n x m matrix, +1, or 0, indicating which entry should be counted
%                    in evalutation
% 
% Outputs:
%         Accu --- 1 x m vector, classification accuracy of each output
%                   dimension
%         Prec --- Precision of each output dimension
%         Recl --- Recall of each output dimension
%         Roc  --- Roc score of each output dimension
%         Roc_mean  --- mean of Roc score
%         Fmac --- macro average of F-values
%         Fmic --- micro average of F-values
%         SqEr --- mean square error

error(nargchk(2, 3, nargin));
if nargin < 3
	nmiss = [];
end
if isempty(nmiss)
	nmiss = ones(size(Ytest));
end

Cpred = sign(Ypred);
Accu = 1- sum(abs((Cpred - Ytest)/2.*nmiss))./sum(nmiss);

overlap = sum(((Cpred + Ytest).*nmiss) == 2);
Prec = overlap./(sum(Cpred.*nmiss == 1)+eps);   % '+ eps' to prevent 'divid by zero'
Recl = overlap./(sum(Ytest.*nmiss == 1)+eps);

m = size(Ytest, 2);                               % number of predictors
ROC = zeros(1, m) + 0.5;                          % set the default value of roc as 0.5
k = min(50, min(sum(nmiss)));                     % number of measured precision vs. 1-specifity pairs. NOTE: large k causes high computational cost.
index = find(sum(nmiss)>=k);
for j = 1 : length(index)
    i = index(j);
    if  isempty(find(Ytest(find(nmiss(:,i)), i) == 1)) == 0
        ROC(i) = roc_curve(Ypred(find(nmiss(:,i)), i), Ytest(find(nmiss(:,i)), i), k);
    else
        ROC(i) = 0.5;
    end
end

Fmac = (2.*Prec.*Recl)./(Prec + Recl + eps);
Fmac = mean(Fmac);

% compute the micro-average of F-value
Recl_micro = sum(overlap)/(sum(sum(Ytest.* nmiss==1))+eps);
Prec_micro = sum(overlap)/(sum(sum(Cpred.* nmiss==1))+eps);
Fmic = (2*Prec_micro*Recl_micro)/(Prec_micro + Recl_micro + eps);

Error = Ypred-Ytest;
Error = Error(find(nmiss));
SqEr = sum(Error.*Error)/sum(sum(nmiss));

Performance = struct( ...
    'Accu', Accu, ... 
    'Prec', Prec, ...
    'Recl', Recl, ...
    'ROC',  ROC, ...
    'ROC_mean', mean(ROC), ...
    'F_macro', Fmac, ...
    'F_micro', Fmic, ...
    'Square_err', sqrt(SqEr)  ...
);

