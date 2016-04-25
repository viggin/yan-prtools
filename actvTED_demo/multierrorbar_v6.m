function [h_bar, h_line] = multierrorbar_v6(X, E, Curve, Marker, Color, XTick, LineWidth, MarkerSize)

% Usage: [h_bar, h_line] = multierrorbar(X, E, Curve, Marker, Color)
% plot multiple curves with errorbars 
% 
% Input:
%         X --- N x M matrix, each row is a curve
%         E --- N x M error matrix, each entry should be nonnegative
%         Curve --- vector of cell strings
%         Marker --- vector of cell strings. Beside those given by plot.m,
%                   '-' is additionally specified, which means a 'null' marker.
%         Color --- vector of cell strings
%         XTick --- must be a ROW vector (matlab v6) or a column vector(v7). 
%                   If not empty, manually set the ticks for x axis
%                   
% 
% Output:
%         h --- a handle, composed by [h(1), h(2)], where the first refers
%         to lines and the second error bars.
% 
% Example:
%          t = [1:20];
%          w = pi./[20:20:400];   % 5 curves
%          X = repmat(w', [1, length(t)]).*repmat(t, [length(w), 1]);
%          Y = sin(X);
%          E = ones(size(Y))*0.05;    % errors
%          figure;
%          [h_bar, h_line] = multierrorbar(Y, E, [], {'-'});   % here '-' means 'null marker'
%          legend(h_line, 'AAA', 'BBB');
%          set(h_line, 'LineWidth', 2, 'MarkerSize', 12);
%          set(gcf, 'DefaultAxesLineWidth',1);
      

error(nargchk(1, 8, nargin));
if nargin < 8
    MarkerSize = [];
end
if isempty(MarkerSize)
    MarkerSize = 12;
end
if nargin < 7
    LineWidth = [];
end
if isempty(LineWidth)
    LineWidth = [];
end
if isempty(LineWidth)
    LineWidth = 2;
end
if nargin < 6
    XTick = [];
end
if nargin < 5
    Color = [];
end 
if isempty(Color)
    Color = {'k', 'b', 'g', 'r', 'c', 'm', 'y'};
end
if nargin < 4
    Marker = [];
end
if isempty(Marker)
    Marker = {'', '.', 'o', 'x', '+', '*', 's', 'd', 'v', '^', '<', '>', 'p', 'h' };
end
if nargin < 3
    Curve = [];
end 
if isempty(Curve)
    Curve = {'-', ':', '-.', '--'};
end

[N, M] = size(X);
if N > length(Curve)*length(Marker)*length(Color)
    error('The rows in X exceed the maximum of (color, marker, curve) combinations!');
end

clf;
set(gca, 'FontSize', 16);
set(gcf, 'DefaultAxesLineWidth',1);
lines = 0; 
h_line = [];
h_bar = [];

for i = 1 : length(Color)
    for j = 1 : length(Marker)
        for k = 1 : length(Curve)
            lines = lines + 1;
            if lines > N
                break;
            end
            if Marker{j} == ''    % stand for 'no marker'!
                if sum(E(lines,:)) == 0     
                    h1 = plot([1:M], X(lines,:), [Color{i}, Curve{k}], 'LineWidth', LineWidth, 'MarkerSize', MarkerSize);   % if no error computed, just plot the curve
                    h_line = [h_line, h1];
                else
                    h1 = errorbar([1:M], X(lines,:), E(lines,:), [Color{i}, Curve{k}]);   
                    %set(h1, 'LineWidth', LineWidth, 'MarkerSize', MarkerSize); h_line = [h_line, h1];      %for matlab7
                    set(h1(2), 'LineWidth', LineWidth, 'MarkerSize', MarkerSize);  h_line = [h_line, h1(2)]; % for matlab 6
                end
            else
                if sum(E(lines,:)) == 0
                    h1 = plot([1:M], X(lines,:), [Color{i}, Marker{j}, Curve{k}], 'LineWidth', LineWidth, 'MarkerSize', MarkerSize);   % if no error computed, just plot the curve
                    h_line = [h_line, h1];
                else
                    h1 = errorbar([1:M], X(lines,:), E(lines,:), [Color{i}, Marker{j}, Curve{k}]); 
                    %set(h1, 'LineWidth', LineWidth, 'MarkerSize', MarkerSize); h_line = [h_line, h1]; %for matlab7
                    set(h1(2), 'LineWidth', LineWidth, 'MarkerSize', MarkerSize); h_line = [h_line, h1(2)]; % for matlab 6
                end
            end
            hold on;
%             h_line = [h_line, h1(2)];
%             h_bar = [h_bar, h1(1)];
        end
    end
end

% set(h_line, 'LineWidth', 2, 'MarkerSize', 12);

if isempty(XTick) == 0
    set(gca, 'XTickMode', 'manual');
    %set(gca, 'XTick', 1:length(XTick));
    xTicks = cellstr(num2str(XTick));
    set(gca, 'XTick', 1:length(XTick), 'XTickLabel', xTicks);
end

V = axis;
margin = (V(2)-V(1))/50;
axis([1-margin, M+margin, V(3), V(4)]);

% 
% 
% set(gca, 'XTickMode', 'manual');
% set(gca, 'XTick', 1:length(C));
% xTicks = cellstr(num2str(C'));
% set(gca, 'XTickLabel', xTicks);
% Xlabel('C (postive example weight)')
% Ylabel('AUC Score');
% title('AUC Score under Various Paramters');
% legend(h_line, {'lambda = 1', 'lambda = 5', 'lambda = 10', 'lambda = 15', 'lambda = 20', 'lambda = 50', 'lambda = 70', 'lambda = 100'}, 4);
