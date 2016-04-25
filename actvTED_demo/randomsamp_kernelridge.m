function accu = randomsamp_kernelridge(X, K, Y, trsize, repeat, lambda, multiclass)

% Usage: accu = randomsamp_kernelridge(X, K, Y, trsize, repeat, lambda, multiclass)

% Random sampling for kernel ridge regression, Since the
% data selection process is independent of labels, we first randomly
% select a sequence of data samples and them simulate the process of 
% learning with more and more seleted examples. 
% Inputs:
%         X --- data matrix, n by d
%         K --- n by n kernel matrix
%         Y --- n by m matrix, either multilabel or multiclass (1-of-c coding)
%         trsize --- a row vector, the sizes of training data. Since we are
%                    simulating a sequential process, evalutation will be
%                    done on each size of training data
%         repeat --- number of repeated active learning experiments
%         lambda --- regularization parameter for kernel regression
%         multiclass --- true if Y is multiclass
% 
% Outputs:
%         accu --- repeat x size(trsize), record every evaluated
%                  classification accuracy

max_size = max(trsize);
N = size(X, 1);
accu = zeros(repeat, length(trsize));

fprintf('Random samplng for kernel ridge regression: \n');

% the learning curve is averaged over several random repeats. 
for i_repeat = 1 : repeat
    fprintf('     Random repeat %d: \n', i_repeat);
    ini_index = classevensamp(Y, 1);                        % initialize training data, to ensure at least one class having one example
    samp_index = [ini_index; incremental_randsamp(N, max_size-length(ini_index), ini_index)];  % then randomly select the remaining training data
    for step = 1 : length(trsize)               % now start to simulate the process of learning with growing size of training data ...
        Ntr = trsize(step);
        Xtr = X(samp_index(1:Ntr), :);
        Ytr = Y(samp_index(1:Ntr), :);
        % perform kernel regression
        Ktr = submatrix(K, samp_index(1:Ntr), samp_index(1:Ntr));
        Ypred = K(:,samp_index(1:Ntr))*inv(Ktr + lambda*eye(Ntr))*Ytr;    % here we set the bias term b = 0
        if multiclass == 1
            Cpred = oneofc_inv(Ypred);
            Ctrue = oneofc_inv(Y);
            accu(i_repeat, step) = sum(Cpred == Ctrue)/N;
        else 
            % if multilabel case ...
            result = multilabel_accu(Ypred, Y);
            accu(i_repeat, step) = mean(result.ROC); 
        end
    end
end
fprintf('\n');
