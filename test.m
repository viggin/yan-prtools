% test the functions in the toolbox
%	Note: the accuracies of different algorithms should not be used to judge
% their effectiveness, since the accuracies depend on dataset, feature
% preprocessing, and algorithm parameters.

close all

%% example of ftProc
%{1
dsName = 'multi-class two gauss'; % see test_genDataset.m for more datasets
[X,Y] = test_genDataset(dsName);
[Xnew, model] = ftProc_pca_tr(X,[],struct('pcaCoef',2)); % PCA
figure,gscatter(Xnew(:,1),Xnew(:,2),Y), title('PCA')

trainIdx = randperm(size(X,1),ceil(size(X,1)*.6));
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
dsName = 'binary-class two gauss'; % see test_genDataset.m for more datasets
[X,Y] = test_genDataset(dsName);
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

%% regression
fprintf('\n>>>>>>regression:\n')
dsName = 'regression synthetic'; % see test_genDataset.m for more datasets
[X,Y] = test_genDataset(dsName);

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

%% feature selection
fprintf('\n>>>>>>feature selection:\n')
dsName = 'classf ionosphere'; % see test_genDataset.m for more datasets
[X,Y] = test_genDataset(dsName);

trainIdx = randperm(size(X,1),ceil(size(X,1)/2));
testIdx = setdiff(1:size(X,1),trainIdx);

cvObj = cvpartition(Y(testIdx),'k',10);
userdata.cvObj = cvObj;
userdata.ft = X(testIdx,:);
userdata.target = Y(testIdx);
if strcmp('dsName','ftSel synthetic')
	nSel = 10;
	maFt = false(1,size(X,2)); % mask for feature indices
	maFt(1:nSel) = true;
	fprintf('ground truth test err: %.4f%%\n', test_getErrRate(maFt,userdata)*100);
else
	nSel = 10;
	maFt = true(1,size(X,2));
	fprintf('all features test err: %.4f%%\n', test_getErrRate(maFt,userdata)*100);
end

mtds = {
	@ftSel_corr, @ftSel_fisher, @ftSel_mrmr,... % filters
	@ftSel_single, @ftSel_sfs,	@ftSel_ga,... % wrappers
	@ftSel_rf,	@ftSel_stepwisefit,	@ftSel_svmrfe_ori, @ftSel_svmrfe_ker,...
	@ftSel_boost % embedded
	};
nMtd = length(mtds);
for iMtd = 1:nMtd
	fprintf([func2str(mtds{iMtd}) ': '])
	% feature selection only on training set
	[ftRank,ftScore] = mtds{iMtd}(X(trainIdx,:),Y(trainIdx));
	fprintf('%d,',ftRank)
	
	maFt = false(1,size(X,2));
	maFt(ftRank(1:min(length(ftRank),nSel))) = true;
	fprintf('\ntest err: %.4f%%\n', test_getErrRate(maFt,userdata)*100);

	fprintf('\n')
end

%% sample selection (active learning)
fprintf('>>>>>>sample selection:\n')
dsName = 'multi-class two gauss'; % see test_genDataset.m for more datasets
[X,Y] = test_genDataset(dsName);

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
	fprintf('\ntest err: %.4f%%\n\n',err*100)
end