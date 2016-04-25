classdef adaboost < handle
	
	properties
		
		wlHandle;
		nIter;
		weakLearners;
		alpha;
		errAll;
		realLen;
		
	end
	
	methods
		
		function a = adaboost(wl,nIter)
			
			a.wlHandle = wl;
			a.nIter = nIter;
			a.alpha = nan(1,nIter);
			a.weakLearners = cell(1,nIter);
			a.errAll = nan(1,nIter);
			a.realLen = 0;
			
		end
		
		function train(a,ft,target)
			
			smpNum = size(ft,1);
			W = ones(smpNum,1)/smpNum;
			ensemble_results = zeros(smpNum,1);
			
			for t = 1:a.nIter
				a.weakLearners{t} = a.wlHandle(false);
				a.weakLearners{t}.train(ft,target,W);
				pred_label1 = a.weakLearners{t}.test(ft); % +1 or -1
				err = sum(W(pred_label1 ~= target));
				
				a.alpha(t) = 1/2 * log((1-err)/max(err,eps));
				W = W.* exp(-a.alpha(t) * target.*pred_label1);
				W = W./sum(W);
				ft(:,a.weakLearners{t}.dim) = NaN;%rand(1,smpNum); % not allow duplicate ft dim sel
				
				% Calculate the current error of the ensemble of weak classifiers
				ensemble_results = ensemble_results + pred_label1*a.alpha(t);
				pred_label = sign(ensemble_results);
				a.errAll(t) = sum(pred_label ~= target)/smpNum;
								
% 				fprintf('%.4f, %d\n',a.errAll(t), a.weakLearners{t}.dim)
				if(a.errAll(t) == 0), break; end
			end
			
			a.realLen = t;
			
		end
		
		function pred_label = test(a,ft,level)
	
			smpNum = size(ft,1);
			ensemble_results = zeros(smpNum,1);
			if ~exist('level','var')
				level = a.realLen;
			end
			for t = 1:level
				ensemble_results = ensemble_results + ...
					a.alpha(t) * a.weakLearners{t}.test(ft);
			end
			pred_label = sign(ensemble_results);
		
		end
		
	end
	
end
