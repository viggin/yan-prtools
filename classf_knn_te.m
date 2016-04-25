function [pred,prob] = classf_knn_te(model,Xtest)
%Classify XTEST based on the trained MODEL
%	MODEL: result of CLASSF_KNN_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	PRED : predicted labels for XTEST
%	PROB : not available.
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

Xtr = model.X;
Ytr = model.Y;
nTest = size(Xtest,1);

% kdtree is sometimes slower than exhaustive
IDX = knnsearch(Xtr,Xtest,'K',model.K,'NSMethod','exhaustive');
nnLbs = Ytr(IDX);
% if knn_k == 1, nnLbs = nnLbs'; end
nnClsCnt = zeros(nTest,model.nClass);
for p = 1:model.nClass
	nnClsCnt(:,p) = sum(nnLbs==p,2);
end

[~,pred] = max(nnClsCnt,[],2);
prob = [];

end