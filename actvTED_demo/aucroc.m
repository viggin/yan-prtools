function auc = aucroc(vx,vy,segnum)
% Calculate the Area Under Curve in a ROC figure
% vx: a vector of each point's x-axis value
% vy: a vector of each point's y-axis value
% segnum: the number of segmentation for the x-axis

intsum = 0;
lengvx = length(vx);
for i=1:segnum 
    currpt = i/segnum;
    % find the nearest point in vx to currpt
    currdist = 1;
    currnear = 0;
    for j=1:lengvx
        currnewdist = abs(currpt-vx(j));
        if currnewdist < currdist
            currnear = j;
            currdist = currnewdist;
        end
    end
    % add the effect of current small range
    intsum = intsum + vy(currnear)/segnum;
end
auc = intsum;