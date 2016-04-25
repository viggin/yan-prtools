function rv = regress_elm_te(model,Xtest)
%Fit XTEST based on the trained MODEL
%	MODEL: result of REGRESS_ELM_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	RV : predicted values for Xtest
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

nTest = size(Xtest,1);

Xtest = Xtest';
tempH_test = model.InputWeight*Xtest;
tempH_test = tempH_test + repmat(model.BiasofHiddenNeurons,1,nTest);
H_test = model.actFun(tempH_test);

rv = H_test' * model.OutputWeight;

end