function [ pred,model,prob ] = classf_( methodName,X,Y,param,Xtest )
%Uniform wrapper of classifiers.
%   A wrapper to call classf_methodName_tr/te. Returns the predicted labels
%   PRED for XTEST and its confidence PROB, the MODEL of the classifier for later
%   test data. The parameters PARAM is optional. PROB may only be available
%   for some methods.
%	METHODNAME is case insensitive.
% 
%   Example:
%		[pred,model,prob] =	classf_('svm',X,Y,struct('c',10),Xtest);
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

if ~exist('param','var')
	param = [];
end

methodName = lower(methodName);
code = sprintf('model=classf_%s_tr(X,Y,param);', methodName);
eval(code);
code = sprintf('[pred,prob]=classf_%s_te(model,Xtest);', methodName);
eval(code);

end

