function rv = regress_rf_te(model,Xtest)
%Fit XTEST based on the trained MODEL
%	MODEL: result of REGRESS_RF_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	RV : predicted values for Xtest
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

addpath RandomForest-v0.02\RF_Reg_C
rv = regRF_predict(Xtest,model);

end
