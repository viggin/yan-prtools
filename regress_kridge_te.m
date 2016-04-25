function [rv] = regress_kridge_te(model,Xtest)
%Fit XTEST based on the trained MODEL
%	MODEL: result of REGRESS_KRIDGE_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	RV : predicted values for Xtest
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

[Xtestz] = ftProc_zscore_te(model.modelz,Xtest);
KXtestz = model.kerFun([ones(size(Xtestz,1),1),Xtestz], model.trXz);
rv = KXtestz * model.b + model.b0;

end