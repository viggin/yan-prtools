function [ftRank,ftScore] = ftSel_svmrfe_ker(ft,label,param)
%Feature ranking using SVM-recursive feature elimination (SVM-RFE).
%	This is the kernel version of SVM-RFE in Guyon "Gene selection 
% for cancer classification using support vector machines". It can handle
% nonlinear SVM models, but are slower than the original SVM-RFE.
%	An correlation bias reduction (CBR) strategy is designed to deal with 
% the problem in SVM-RFE when lots of highly correlated features exist.
%   Only tested on binary-class cases. For multi-class cases, we simply
% add the feature weights of each binary-class subproblems. This strategy
% hasn't been verified.
%
%	FT:	feature matrix, each row is a sample. Better be scaled, such as
% zero-mean and unit-variance
%	LABEL:	column vector of class labels of FT
%	PARAM: struct of parameters. The beginning part of this code (before 
% defParam) explains each parameter, and also sets the default parameters. 
% You can change parameter p to x by setting PARAM.p = x. For parameters
% that are not set, default values will be used. If the number of parameters
% scares you, just tune the important parameters:
%	PARAM.kerType, rfeC, and rfeG: the parameters for the kernel used in
% SVM training. Important, must be consistent with the data. See
% LIBSVM toolbox for detailed meanings.
%	PARAM.useCBR: whether or not use the CBR strategy. If lots of highly
% correlated features exist, use it may be better.
%	PARAM.Rth: correlation coef threshold for highly corr features.
%
%	FTRANK: rank of the features (most important first)
%	FTSCORE: a continuous score of each feature in FTRANK in each iteration.
% Just for logging purpose in this function.
%
%	Dependency: libsvm toolbox
% 
%	Please refer to Ke Yan et al., Feature selection and analysis on correlated
% gas sensor data with recursive feature elimination, Sensors and Actuators
% B: Chemical, 2015
%	Thanks to the spider toolbox and Rakotomamonjy's SVM-KM toolbox


%% default parameters, can be changed by specifying the field in PARAM
% parameters for general SVM-RFE
kerType = 2; % kernel type, see libsvm. linear: 0; rbf:2
rfeC = 2^0; % parameter C in SVM training
rfeG = 2^-6; % parameter g in SVM training
nStopChunk = 60; % when number of features is less than this num, start 
% removing one-by-one for precision. if set to inf, only remove one-by-one,
% accurate but slow.
useApproxAlpha = true; % whether use approximate alpha. See Rakotomamonjy's paper
%	"Variable selection using SVM based criteria". Set to true is safe and faster
rmvRatio = .5; % ratio of num of removed features before stopChunk

% parameters for CBR
useCBR = true; % whether or not use CBR
nOutCorrMax = 1; % max num of highly correlated features that can be removed
%	each iteration, when no feature highly corr with them is still kept. See our paper
Rth = .9; % corrcoef threshold for highly corr features

defParam % handle the parameters

%% prepare
nFt = size(ft,2);
ftOut = find(std(ft)<=1e-5); % indices of removed features. First, remove constant features
ftIn = setdiff(1:nFt,ftOut); % indices of survived features
ftScore = [];

if useCBR
	R_all = abs(corrcoef(ft));%abs(corr(ft,'type','spearman'));%
end

kerOpt.C = rfeC;
kerOpt.g = rfeG;
kerOpt.type = kerType;

%% run
while ~isempty(ftIn)
	
	nFtIn = length(ftIn);
	[supVec,alpha_signed] = trainSVM(ft(:,ftIn),label,kerOpt);
	
	nSv = size(supVec,1);
	svInProd = supVec*supVec';
	svSqr = sum(supVec.^2,2);
	kerMatAll0 = repmat(svSqr,1,nSv) + repmat(svSqr',nSv,1) - 2*svInProd;
	w2_allIn = trace(alpha_signed' * exp(-kerOpt.g * kerMatAll0) * alpha_signed);
	% trace is used to add up the feature weights of each binary-class 
	% subproblems. This strategy hasn't been verified.
	
	w2_in = zeros(1,nFtIn);
	if useApproxAlpha
		
		for iFtIn = 1:nFtIn
			supVecP = supVec(:,iFtIn);
			% use the method in spider toolbox to compute the weight for
			% each feature p.
			kerMatP0 = (repmat(supVecP,1,nSv) - repmat(supVecP',nSv,1)).^2;
			kerMatRemoveP = exp(-kerOpt.g * (kerMatAll0 - kerMatP0));
			
			% the approximate margin when feature p is removed
			% it can be proved that w2=alpha_signed'*K*alpha_signed=
			% sum(alpha) when alpha is approximated
			w2_in(iFtIn) = trace(alpha_signed'*kerMatRemoveP*alpha_signed);
			% 				if rem(p,100)==0, fprintf(','); end
		end
		
	else % use precise alpha, much slower
		
		for iFtIn = 1:nFtIn
			
			ftInWoP = ftIn;
			ftInWoP(iFtIn) = [];
			[supVecWoP,alpha_signedWoP] = trainSVM(ft(:,ftInWoP),label,kerOpt);
			svNumWoP = size(supVecWoP,1);
			svInProdWoP = supVecWoP*supVecWoP';
			svSqrWoP = sum(supVecWoP.^2,2);
			kerMatAll0WoP = repmat(svSqrWoP,1,svNumWoP) + ...
				repmat(svSqrWoP',svNumWoP,1) - 2*svInProdWoP;
			
			% the exact margin when feature p is removed
			% it can be proved that w2=alpha_signed'*K*alpha_signed=
			% sum(alpha) when alpha is not accurate
			w2_in(iFtIn) = trace(alpha_signedWoP' * exp(-kerOpt.g * kerMatAll0WoP) *...
				alpha_signedWoP);
			% 				if rem(iFtIn,10)==0, fprintf('%d,',iFtIn/10); end
			
		end
	end
	
	% criteria for each ft in ftIn, the larger the more important
	criteria = (w2_allIn-w2_in)/2; % adding abs is not good
	% 		figure,plot(ftIn,criteria,'.-')
	[criteria_sort,idx] = sort(criteria,'ascend');
	
	% for logging purpose
	w2_tmp = nan(1,nFt);
	w2_tmp(ftIn) = criteria;
	ftScore = [ftScore;w2_tmp];
	
	% how many features to remove
	if nFtIn > nStopChunk
		nRemove = floor(nFtIn*rmvRatio);
		if nFtIn-nRemove < nStopChunk
			nRemove = nFtIn-nStopChunk;
		end
	else
		nRemove = 1;
	end
	
	ftOutCur = idx(1:nRemove); % to be removed
	FocRealIdx = ftIn(ftOutCur); % the real ft indices
	
	%% CBR
	if useCBR && Rth < 1 && nRemove > 1
		ftInTemp = ftIn;
		ftInTemp(ftOutCur) = [];
		
		no_rmv = [];
		% rescue some features
		for iFtOut = nRemove:-1:1
			inSimilarNum = nnz(R_all(FocRealIdx(iFtOut),ftInTemp) > Rth);
			outSimilarNum = nnz(R_all(FocRealIdx(iFtOut),FocRealIdx) > Rth);
			if inSimilarNum < 1 && outSimilarNum > nOutCorrMax
				no_rmv = [no_rmv iFtOut];
				ftInTemp = [ftInTemp FocRealIdx(iFtOut)];
			end
		end
		ftOutCur(no_rmv) = [];
		FocRealIdx = ftIn(ftOutCur); % the real ft indices
		
	end % if useCBR && Rth < 1 && nRemove > 1
	
	ftOut = [ftOut,FocRealIdx];
	ftIn(ftOutCur) = [];
	
	if nRemove>1, fprintf('%d|',length(ftIn)); end
end % while ~isempty(ftIn)

ftRank = fliplr(ftOut); % least important ft in the end
ftScore = ftScore(:,ftRank); % just for logging. sorted according to ftRank

end


function [supVec,alpha_signed] = trainSVM(X,Y,kerOpt)
% use libsvm to find the support vectors and alphas

type = sprintf('-s 0 -t %d -c %f -g %f -q',kerOpt.type,kerOpt.C,kerOpt.g);
model = svmtrain(Y, X, type);
if isempty(model) || sum(model.nSV) == 0
	error('libSVM cannot be trained properly. Please check your data')
end
supVec = full(model.SVs);
alpha_signed = model.sv_coef;
% svIdxs = model.sv_indices; % older versions of libSVM don't have it

end