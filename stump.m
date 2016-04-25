classdef stump < handle
%Stump weak learner for AdaBoost
%	If there is a feature that holds the same value in many samples, 
% error will occur.

	properties
		
		dim;
		direc;
		th;
		err;
		trainMin;
		trainMax;
		
	end
	
	methods
		
		function a = stump(~)
			
		end
		
		function train(a, ft,target,W)
			
			% best threshold searching method is according to
			% Viola's "Robust Real-Time Face Detection" in IJCV,2004
			
			[smpNum,ftNum] = size(ft);
			coefsPerDim = zeros(ftNum,3);
			% each row: threshold; direction; weighted error
			
			for p = 1:ftNum
% 				if p==459
% 					tea;
% 				end
				ft1 = ft(:,p);
				if any(isnan(ft1))
					coefsPerDim(p,3) = 1; % max error
					continue; 
				end
				[ft1_sorted, I1] = sort(ft1,'ascend');
				target_sorted = target(I1);
				W_sorted = W(I1);
				pos = (target_sorted==1);
				neg = (target_sorted==-1);
				
				W_split = zeros(2,smpNum);
				W_split(1,pos) = W_sorted(pos);
				W_split(2,neg) = W_sorted(neg);
				W_cum = cumsum(W_split,2);
				W_total = repmat(W_cum(:,end),1,smpNum);
				errMat = W_cum+flipud(W_total)-flipud(W_cum);
				% errMat(1,k):weighted error if classify all samples > ft1_sorted(k)
				%	as positive class, and <= ft1_sorted(k) as negative class
				% errMat(2,k):weighted error if classify all samples > ft1_sorted(k)
				%	as negative class, and <= ft1_sorted(k) as positive class
				
				[err1, loc] = min(errMat(:));
				[direc1,loc] = ind2sub(size(errMat),loc);
				if loc < smpNum % threshold: middle between 2 samples
					coefsPerDim(p,1) = mean(ft1_sorted(loc:loc+1));
				else % if best th is larger than the largest sample(possible?)
					coefsPerDim(p,1) = ft1_sorted(loc)+...
						(ft1_sorted(loc)-ft1_sorted(1))/(smpNum-1)/2;
				end
				coefsPerDim(p,2) = (direc1==2)*2-1;
				coefsPerDim(p,3) = err1;
				
			end
			
			[a.err, a.dim] = min(coefsPerDim(:,3));
			a.th = coefsPerDim(a.dim,1);
			a.direc = coefsPerDim(a.dim,2);
			a.trainMin = min(ft(:,a.dim));
			a.trainMax = max(ft(:,a.dim));
			
		end		
		
		function Y = test(a, X)
			
			Y = double((a.direc * X(:,a.dim) < a.direc * a.th));
			% according to Viola's "Robust Real-Time Face Detection" in IJCV,2004
			
			Y(Y==0) = -1;
			
		end
		
	end
	
end