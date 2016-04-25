function [X, Y] = drawdata()

% Usage: [X, Y] = drawdata()
% Manually draw a toy data set
% X --- Nx2 data matrix
% Y --- Nx1 labels
% click left mouse button to draw a new point
% click right mouse button to shift to a new class
% double clike right mouse to stop and quit



Color = {'k', 'b', 'g', 'r', 'm'};
Marker = {'.', '*', 'o', '+'};

f = figure;
set(f, 'Position',  [410 214 740 620]); 
set(gca, 'Fontsize', 16);
plot(0, 0, 'w.');
title('Draw 2-D data by clicking mouse');
xlabel('Double-click right button to stop.');
axis([-1 1 -1 1])
hold on
% Initially, the list of points is empty.
X = [];
Y = [];
n = 0;
% % Loop, picking up the points.
% disp('Left mouse button picks points.')
% disp('Right mouse button picks last point.')

class_label = 0;
stop = 0;

class = 0;
while stop  == 0
    newX = [];
    newY = [];
    class_label = class_label + 1;
    n = 0;
    but = 1;
    class = class + 1;
    colorindex = mod(class, length(Color));
    if colorindex == 0
        colorindex = length(Color);
    end
    markerindex = floor(class/length(Color))+1;
    data_marker = [Color{colorindex}, Marker{markerindex}];
    while but == 1
        [xi,yi,but] = ginput(1);
        if but == 1
            plot(xi,yi, data_marker, 'MarkerSize', 16, 'LineWidth', 2);
            n = n+1;
            newX(n, :) = [xi, yi];
            newY(n, 1) = class_label;
        end
    end
    X = [X; newX];
    Y = [Y; newY];
    if isempty(newX) == 1
        stop = 1;
    end
end

% but = 1;
% while but == 1
%     [xi,yi,but] = ginput(1);
%     if but == 1
%         plot(xi,yi,'bx')
%         n = n+1;
%         X(n,:) = [xi, yi, 1];
%     end
% end
% 
% hold off;
% x = X(:,1:2);
% y = X(:,3);