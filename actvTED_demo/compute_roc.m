function [auc,tpr,err,pic,auc_cut] = compute_roc(func, label)

% ROC curve computation
% input: predictive probability --- func
%        target labels          --- label
% output: area under ROC curve

solu=1001 ;

[m,n] = size (func) ;
[p,q] = size (label) ;
if (m~=p | n~=1 | q~=1)
    error('dimensions are inconsistent.') ;
end

auc=0 ;
fn=0 ; %false negative
tn=0 ; %true negative
cn = sum(label<0.5) ;
cp = sum(label>=0.5) ;
tp=cp; %true positive
fp=cn ; %false positive
if (cn+cp~=m)
    error('labels are wrong.') ;
end
if (cn<1 | cp<1)
    disp('one class problem.') ;
    return ;
end

%
pic = zeros(solu,1) ;
ind = 0 ;

sensitivity = tn/cn ;
specificity = fn/cp ;
step = sensitivity ;
roc(1,1) = specificity ;
roc(1,2) = sensitivity ;
% sort func
[sf,index] = sort(func) ;
sl = label(index) ;

for i=1:m
    if (sl(i)<0.5)
        tn = tn + 1 ;
        fp = fp - 1 ;
        sensitivity = tn/cn ;
    else
        fn = fn + 1 ;
        tp = tp - 1 ;
        specificity = fn/cp ;
        newindex = ceil(specificity*solu) ;
        for k=ind+1:newindex
            pic(k)= sensitivity ;
        end
        ind = newindex-1 ;
    end
    auc = auc + (sensitivity-step)*(1-specificity) ;
    tpr(i) = tp/(tp+fn) ;
    err(i) = ( tp + tn ) / m ;
    if (fn==ceil(cp/10))
    	auc_cut = (auc-sensitivity*(1-specificity))/ceil(cp/10)*cp ;
    end
    step = sensitivity ;
    roc(i+1,1) = specificity ;
    roc(i+1,2) = sensitivity ;
end

[a,b]=min(err);
disp(sprintf('true positive rate %f',tpr(b))) ;
disp(sprintf('accuracy %f',1-a)) ;

figure(1)
hold on
box on
plot(roc(:,1),roc(:,2))
figure(2)
hold on
plot(0:0.001:1,pic,'r')
box on
return ;

