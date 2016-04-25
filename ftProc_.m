function [ XNew,model,XtestNew ] = ftProc_( methodName,X,Y,param,Xtest )
%Uniform wrapper of feature processing.
%	[ XNew,model,XtestNew ] = ftProc( methodName,X,Y,param,Xtest )
%   A wrapper to call ftProc_methodName_tr/te. Returns the processed
%   data XNEW, the MODEL of the processing method (if exists) for later
%   test data. The input label Y, parameters PARAM, test data XTEST, and
%   the output XTESTNEW are all optional.
%	METHODNAME is case insensitive.
% 
%   Example:
%		[X,~,Xtest] =	ftProc_('mat2ftvec',X,[],[],Xtest);
%		[XPca,model,XtestPca] =	ftProc_('pca',X,[],struct('pcaCoef',2),Xtest);
% 
%	Ke YAN, 2016, Tsinghua Univ. http://yanke23.com, xjed09@gmail.com

if ~exist('param','var')
	param = [];
end
if ~exist('Y','var')
	Y = [];
end

methodName = lower(methodName);
if exist(['ftProc_' methodName],'file') == 2 % no tr and te
	code = sprintf('XNew=ftProc_%s(X,Y,param);', methodName);
	eval(code);
	if ~exist('Xtest','var')
		code = sprintf('XtestNew=ftProc_%s(Xtest,Y,param);', methodName);
		eval(code);
	end
	
else % have tr and te
	code = sprintf('[XNew,model]=ftProc_%s_tr(X,Y,param);', methodName);
	eval(code);
	if exist('Xtest','var')
		code = sprintf('XtestNew=ftProc_%s_te(model,Xtest);', methodName);
		eval(code);
	end
end

end

