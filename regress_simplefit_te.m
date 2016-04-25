function [rv] = regress_simplefit_te(model,Xtest)
%Fit XTEST based on the trained MODEL
%	MODEL: result of REGRESS_SIMPLEFIT_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	RV : predicted values for Xtest
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

if model.method == 1 % robust fit, univariate or multivariate
	rv = [ones(testNum,1),Xtest]*model.coefs;
elseif model.method == 2 % univariate polynomial fit
	rv = polyval(model.coefs,Xtest);
elseif model.method == 3 % general method. See doc for LinearModel.
	rv = predict(model.coefs,Xtest);
end

end