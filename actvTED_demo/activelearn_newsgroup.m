randseed = 6;
rand('seed', randseed);
randn('seed', randseed);

%*************************************************************************
% Load the newsgroup data, which contains 3970 documents with 4 exclusive
% classes   
%    Inputs:  X   3970x8014
%    Labels:  Y   3970x4  (+1 or -1, 1-of-c coding)
%*************************************************************************
load newsgroup_4c_normalized.mat;
K = full(X*X');                      % compute the kernel matrix, 
                                     % here we use linear kernel 

%*************************************************************************
% Specify the setting of learning curves
%*************************************************************************
maxtrsize = 100;                     % maximum size of labeled samples
repeat = 5;                          % number of random repeats
trsize = [10:10:maxtrsize];          % step size of learning curve

%*************************************************************************
%  Run different active learning methods to obtain learning curves of each
%    1. actveig_kreg:       eigen sampling + kernel regression
%    2. actvrand_kreg:      random sampling + kernel regression
%    3. actvuncert_kreg:    uncertainty sampling + kernel regression
%*************************************************************************
accu_mean = [];                      % initialize mean of learning curves
accu_std = [];                       % initialize std of learning curves
lambda = 0.01;                       % regularization weight for kernel regression
multiclass = 0;                      % set the problem as a set of two-class problems

rand('seed', randseed);
randn('seed', randseed);
repeat = 2;
accu = transdesign_kernelridge(X, K, Y, trsize, repeat, lambda, multiclass);
accu_mean = [accu_mean; mean(accu)]; accu_std = [accu_std; std(accu)]; 

% repeat = 1;
% accu = actvaopt_kreg(X, K, Y, trsize, repeat, lambda, multiclass);
% accu_mean = [accu_mean; mean(accu)]; 

rand('seed', randseed);
randn('seed', randseed);
accu = randomsamp_kernelridge(X, K, Y, trsize, repeat, lambda, multiclass);
accu_mean = [accu_mean; mean(accu)]; accu_std = [accu_std; std(accu)]; 

% accu = actvuncert_kreg(X, K, Y, trsize, repeat, lambda, multiclass);
% accu_mean = [accu_mean; mean(accu)]; accu_std = [accu_std; std(accu)]; 


%*************************************************************************
%  Plot learning curves with mean and errorbar
%*************************************************************************
figure;
[h_bar, h_line] = multierrorbar(accu_mean, accu_std, [], {''}, [], trsize', [], 5);
legend(h_line, 'Transductive Experimental Design', 'Random Sampling');
xlabel('# of Training Examples');
ylabel('AUC');


