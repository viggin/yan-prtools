function [rv] = regress_step_te(model,Xtest)
%Fit XTEST based on the trained MODEL
%	MODEL: result of REGRESS_STEP_TR.
%	XTEST: a matrix, each row is a sample.
% Return:
%	RV : predicted values for Xtest
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

rv = Xtest(:,model.inmodel)*model.b(model.inmodel)+model.stats.intercept;

end
