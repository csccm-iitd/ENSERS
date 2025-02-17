% Save data in .mat file

clear all; close all; clc
addpath(genpath('./'))

%% data parameters
idxLs = 425:449;

dt = 0.2;
timeGrid = (0:length(idxLs)-1)*dt;

% data domain
xRange = [1.5, 5.5]; yRange = [-2, 2]; d = 0.2;
% little wider domain
xRangeWide = [xRange(1)-d, xRange(2)+d]; yRangeWide = [yRange(1)-d, yRange(2)+d]; 

% cartesian grid for interpolation
nx = 64; ny = 64;    
x = linspace(xRange(1), xRange(2), nx);
y = linspace(yRange(1), yRange(2), ny);
[gridX, gridY] = meshgrid(x, y);
gridX = gridX'; gridY = gridY';


%% load and process data

for i=1:length(idxLs)
    disp(i)
    
    % load data
    CFD_raw = importdata(sprintf('data_%d.csv', idxLs(i)));
    
    % change column headers
    new_names = {'Points0', 'Points1', 'Points2','u0', 'u1', 'u2', ...
        'p', 'vort0', 'vort1', 'vort2'};    
    for j=1:10
        disp(CFD_raw.colheaders(j));
        CFD_raw.colheaders(j) = new_names(j);
        disp(CFD_raw.colheaders(j));
    end
    
    % convert raw array to table
    variables_names = arrayfun(@(str) strrep(str, '_', '_'), CFD_raw.colheaders);
    CFD_data = array2table(CFD_raw.data, 'VariableNames', variables_names); 
    
    % delete data outside wider domain
    del_Y = (CFD_data.Points1 < yRangeWide(1)) | (CFD_data.Points1 > yRangeWide(2));
    CFD_data(del_Y,:) = [];
    del_X = (CFD_data.Points0 < xRangeWide(1)) | (CFD_data.Points0 > xRangeWide(2));
    CFD_data(del_X,:) = [];   
    
    % interpolate data to grid
    Fu0 = scatteredInterpolant(CFD_data.Points0, CFD_data.Points1, CFD_data.u0);
    Fu1 = scatteredInterpolant(CFD_data.Points0, CFD_data.Points1, CFD_data.u1);
    Fp = scatteredInterpolant(CFD_data.Points0, CFD_data.Points1, CFD_data.p);
    
    u0 = Fu0(gridX(:), gridY(:));
    u1 = Fu1(gridX(:), gridY(:));
    p = Fp(gridX(:), gridY(:));
    
%     plotImg(u1, ny, nx)
    flowData(i,:, :) = [u0, u1, p]';  % (3, 4096)
end


%% save data
save('flowData.mat', 'flowData')


%% save data parameters

% u--m/s
% p--m2/s2
% nu--m2/s--0.005;
% writeInterval    10;
% dt--0.02

dataParams = struct('save_dt', dt, ...
                    'dx', x(2)-x(1), ...
                    'numtimeStep', length(idxLs), ...
                    'timeGrid', timeGrid, ...
                    'imDim', [64, 64], ...
                    'Re', 200, ...
                    'nu', 0.005);

% Convert to JSON text
jsonText2 = jsonencode(dataParams);
% Write to a json file
fid = fopen('dataParams.json', 'w');
fprintf(fid, '%s', jsonText2);
fclose(fid);



function plotImg(u0, ny, nx)
    img = reshape(u0, [ny nx]);
    figure();
    imagesc(img);
end
