% test the functions in the toolbox
%	Note: the accuracies of different algorithms should not be used to judge
% their effectiveness, since the accuracies depend on dataset, feature
% preprocessing, and algorithm parameters.

%% data for ftProc and classfication
close all

% data 1
% load ionosphere
% Y = (cell2mat(Y)=='b')+1; % text label to number

% data 2, binary classes
% d = 100; n = 50; s = 10;
% X = [mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n);
% 	mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n)];
% Y = [ones(n*2,1);ones(n*2,1)*2];

% data 3, multi-class
d = 100; n = 50; s = 10;
X = [mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n);
	mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n);
	mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n)];
Y = [ones(n*2,1);ones(n*2,1)*2;ones(n*2,1)*3];

%% example of ftProc
%{1
[Xnew, model] = ftProc_pca_tr(X,[],struct('pcaCoef',2)); % PCA
figure,gscatter(Xnew(:,1),Xnew(:,2),Y), title('PCA')

trainIdx = randperm(size(X,1),ceil(size(X,1)/2));
testIdx = setdiff(1:size(X,1),trainIdx);
[XtrNew, model] = ftProc_lda_tr(X(trainIdx,:),Y(trainIdx),...
	struct('postLdaDim',2)); % LDA
[XteNew] = ftProc_lda_te(model,X(testIdx,:));
figure,hold on
gscatter(XtrNew(:,1),XtrNew(:,2),Y(trainIdx),'bbb','.ox')
gscatter(XteNew(:,1),XteNew(:,2),Y(testIdx),'rrr','.ox')
title('LDA')
%}

%% classfication
%{1
fprintf('\n>>>>>>classfication:\n')
cvObj = cvpartition(Y,'k',10);
mtdNames = {'svm','knn','lr','ann','boost',...
	'elm','rf','gauss','softmax','tree'};
% params = {struct('t',2,'c',1),struct('K',7)};

nMtd = length(mtdNames);
errTable = cell(nMtd,2);
errTable(:,1) = mtdNames;
for iMtd = 1:nMtd
	if any(Y>2) && strcmp(mtdNames{iMtd},'boost')
		continue
	end
	
	fprintf(mtdNames{iMtd})
	pred = nan(length(Y),1);
	
	for iCv = 1:cvObj.NumTestSets
		trainIdx = cvObj.training(iCv);
		testIdx = cvObj.test(iCv);
		Xtr = X(trainIdx,:); Xte = X(testIdx,:);
		[pred(testIdx),model,prob ] = classf_(mtdNames{iMtd},...
			Xtr,Y(trainIdx),[],Xte);
	end
	errTable{iMtd,2} = nnz(pred~=Y)/length(Y);
	fprintf(', test err: %.4f%%\n',errTable{iMtd,2}*100)
end
%}

%% feature selection
fprintf('\n>>>>>>feature selection:\n')

useSyntheticData = 0;
if useSyntheticData
	d = 10; n = 100; s = 1;
	X = randn(n,d); % useful features 1:d
	Y = double((X*rand(d,1)+rand(n,1)*s)>0)+1;
	X = [X,randn(n,d)]; % noise features d+1:d*2
else
	load ionosphere
	Y = (cell2mat(Y)=='b')+1; % text label to number
	d = 5;
end

trainIdx = randperm(size(X,1),ceil(size(X,1)/2));
testIdx = setdiff(1:size(X,1),trainIdx);

cvObj = cvpartition(Y(testIdx),'k',10);
userdata.cvObj = cvObj;
userdata.ft = X(testIdx,:);
userdata.target = Y(testIdx);
if useSyntheticData
	x = false(1,d*2);
	x(1:d) = true;
	fprintf('ground truth test err: %.4f%%\n', getErrRate_example(x,userdata)*100);
else
	x = true(1,size(X,2));
	fprintf('all features test err: %.4f%%\n', getErrRate_example(x,userdata)*100);
end

mtds = {
	@ftSel_corr, @ftSel_fisher, @ftSel_ga, @ftSel_mrmr, @ftSel_rf,...
	@ftSel_sfs,	@ftSel_single, @ftSel_stepwisefit,...
	@ftSel_svmrfe_ori, @ftSel_svmrfe_ker,...
	@ftSel_boost
	};
nMtd = length(mtds);
for iMtd = 1:nMtd
	fprintf([func2str(mtds{iMtd}) ': '])
	% feature selection only on training set
	[ftRank,ftScore] = mtds{iMtd}(X(trainIdx,:),Y(trainIdx));
	fprintf('%d,',ftRank)
	
	x = false(1,d*2);
	x(ftRank(1:min(length(ftRank),d))) = true;
	fprintf('\ntest err: %.4f%%\n', getErrRate_example(x,userdata)*100);

	fprintf('\n')
end

%% regression
fprintf('\n>>>>>>regression:\n')

useSyntheticData = 1;
if useSyntheticData
	d = 10; n = 100; s = .1;
	X = randn(n,d); % useful features 1:d
	Y = X*rand(d,1)+randn(n,1)*s;
	X = [X,randn(n,2)]; % noise features
else
	load spectra
	Y = octane;
	X = NIR;
end

cvObj = cvpartition(size(X,1),'k',5);
mtdNames = {...
	'svr','ridge','kridge','lasso','pls',...
	'elm','rf','simplefit','step','ann'};
% params = {struct('t',2,'c',1),struct('K',7)};

nMtd = length(mtdNames);
errTable = cell(nMtd,2);
errTable(:,1) = mtdNames;
for iMtd = 1:nMtd
	if any(Y>2) && strcmp(mtdNames{iMtd},'boost')
		continue
	end
	
	fprintf(mtdNames{iMtd})
	rv = nan(length(Y),1);
	
	for iCv = 1:cvObj.NumTestSets
		trainIdx = cvObj.training(iCv);
		testIdx = cvObj.test(iCv);
		Xtr = X(trainIdx,:); Xte = X(testIdx,:);
		[rv(testIdx),model] = regress_(mtdNames{iMtd},...
			Xtr,Y(trainIdx),[],Xte);
	end
	errTable{iMtd,2} = rms(Y-rv);
	fprintf(', test err: %.4f\n',errTable{iMtd,2})
end

%% sample selection (active learning)
useSyntheticData = 1;
if useSyntheticData
	d = 100; n = 50; s = 10;
	X = [mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n);
		mvnrnd(randn(1,d),eye(d)*s,n);mvnrnd(randn(1,d),eye(d)*s,n)];
	Y = [ones(n*2,1);ones(n*2,1)*2];
else
	load ionosphere
	Y = (cell2mat(Y)=='b')+1; % text label to number
	d = 5;
end

fprintf('\n>>>>>>sample selection:\n')
mtds = {@smpSel_ks,@smpSel_ted,@smpSel_llr,@smpSel_cluster};
nMtd = length(mtds);
nSel = 20;

candidateIdx = randperm(size(X,1),ceil(size(X,1)/2));
testIdx = setdiff(1:size(X,1),trainIdx);
[pred] = classf_('lr',X(candidateIdx,:),Y(candidateIdx),[],X(testIdx,:));
err = nnz(pred~=Y(testIdx))/length(pred);
fprintf('train with all candidate samples, test err=%.4f%%\n',err*100)
	
for iMtd = 1:nMtd
	fprintf([func2str(mtds{iMtd}) ': '])
	smpList = mtds{iMtd}(X(candidateIdx,:),nSel);
	fprintf('%d,',smpList)
	
	trIdx = candidateIdx(smpList);
	[pred] = classf_('lr',X(trIdx,:),Y(trIdx),[],X(testIdx,:));
	err = nnz(pred~=Y(testIdx))/length(pred);
	fprintf('\ntest err=%.4f%%\n\n',err*100)
end