function [ rv,model ] = regress_( methodName,X,Y,param,Xtest )
%Uniform wrapper of regressors.
%   A wrapper to call regress_methodName_tr/te. Returns the predicted labels
%   RV for XTEST and the MODEL of the classifier for later test data. The 
% parameters PARAM is optional.
%	METHODNAME is case insensitive.
% 
%   Example:
%		[rv,model] = regress_('svr',X,Y,struct('c',10),Xtest);
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

if ~exist('param','var')
	param = [];
end

methodName = lower(methodName);
code = sprintf('model=regress_%s_tr(X,Y,param);', methodName);
eval(code);
code = sprintf('rv=regress_%s_te(model,Xtest);', methodName);
eval(code);

end

