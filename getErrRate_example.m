function err = getErrRate_example(x,userdata)
%Sample code for computing the error rate of a feature group, used in wrapper 
% feature selection methods like FTSEL_GA, FTSEL_SFS, FTSEL_SINGLEFTPRED

ftIdx = find(x);
nCv = userdata.cvObj.NumTestSets;
nSmp = size(userdata.ft,1);
pred_label = nan(nSmp,1);

for cv = 1:nCv
	
	trIdx = userdata.cvObj.training(cv);
	teIdx = userdata.cvObj.test(cv);
	ft_train = userdata.ft(trIdx,ftIdx); ft_test = userdata.ft(teIdx,ftIdx);
	pred_label(teIdx) = classf_('lr', ft_train, userdata.target(trIdx),[],...
		ft_test);
end

err = nnz(pred_label~=userdata.target)/nSmp;

end