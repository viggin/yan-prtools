function [ftSubset,ftScore] = ftSel_ga(ft,target,param)
%Feature selection using the genetic algorithm in Matlab
%	FT : sample matrix, each row is a sample.
%	TARGET : a column vector, depend on the wrapper prediction algorithm
% used.
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used.
% Return:
%	FTSUBSET: unordered selected features
%	FTSCORE: not returned.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

% see the doc of ga for the meaning of the possible parameters
% the accuracy can be improved by enlarging popSize, maxIter,
% stallGenLimit, and maybe increasing mutRate
mutRate = .3;
popSize = 20;
maxIter = 3;
stallGenLimit = 2;
tolFun = eps;
getErrRateFunc = @getErrRate_example; % user-defined function for computing the score
								  % of each chromosome
nCv = 5; % the number of cross-validation when computing the score

defParam

userdata.ft = ft;
userdata.target = target;

nFt = size(ft,2);

% set cvObj here if you wish each evaluation uses the same cvObj
cvObj = cvpartition(target,'k',nCv);
userdata.cvObj = cvObj;

fitFn = @(x)getErrRateFunc(x,userdata);

newopt = struct('Display','iter','FitnessLimit',0,...
	'PlotFcns',[],'PopulationType','bitstring',...
	'PopulationSize',popSize,'Generations',maxIter,...
	'CreationFcn',@gacreationuniform,...
	'CrossoverFcn',@crossoverscattered,...
	'StallGenLimit',stallGenLimit,...
	'TolFun',tolFun,...
	'UseParallel','always');

options = gaoptimset(gaoptimset,newopt);
% MutationFcn must be set here, or mutRate won't be set
options = gaoptimset(options,'MutationFcn',{@mutationuniform,mutRate});

tic
[ftSubset,fval,exitflag] = ...
	ga(fitFn,nFt,[],[],[],[],[],[],[],[],options);
toc
ftSubset = find(ftSubset);
ftScore = [];

end
