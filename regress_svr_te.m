function [rv] = regress_svr_te(model,Xtest)
%Fit XTEST based on the trained MODEL
%	MODEL: result of REGRESS_SVR_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	RV : predicted values for Xtest
% 
%	Dependency: libsvm toolbox
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

[~,~, rv] = svmpredict(zeros(size(Xtest,1),1), Xtest, model);

end
