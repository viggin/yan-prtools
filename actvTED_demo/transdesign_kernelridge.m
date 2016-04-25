function accu = transdesign_kernelridge(X, K, Y, trsize, repeat, lambda, multiclass)

% Usage: accu = transdesign_kernelridge(X, K, Y, trsize, repeat, lambda, multiclass)

% Simulation of transductive experimental design for kernel ridge regression. Since the
% data selection process is independent of labels, we first sequentially
% select a sequence of data samples and them simulate the process of 
% learning with more and more seleted examples. 

% Inputs:
%         X --- data matrix, n by d
%         K --- n by n kernel matrix of X
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

fprintf('Active learning via sequential transductive experimental design \n');

% the learning curve is averaged over several random repeats. 
for i_repeat = 1 : repeat
    fprintf('     Random repeat %d: ', i_repeat );
%     ini_index = classevensamp(Y, 1);            % initialize training data, to ensure at least one class having one example
    candidate_index = randsamp(size(K,1), round(size(K,1)*0.5));          % restrict the candidate set as a random subset, to improve the efficiency
    samp_index = transdesign_sq(K, max_size, lambda, candidate_index);    % sequential data selection by transductive experimental design
%     samp_index = [ini_index; samp_index];       % indices of selected data, in the sequencial order of being selected
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
