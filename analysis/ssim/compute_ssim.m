% Copyright (C) 2020 ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland
%
%     Multimedia Signal Processing Group (MMSPG)
%
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
%
% Author:
%   Evangelos Alexiou (evangelos.alexiou@epfl.ch)
%
% Reference:
%   E. Alexiou and T. Ebrahimi, "Towards a Point Cloud Structural
%   Similarity Metric," 2020 IEEE International Conference on Multimedia &
%   Expo Workshops (ICMEW), London, United Kingdom, 2020, pp. 1-6.
%
%
% This is a simple script that serves as an example of a main. The function
%   "pc_ssim" takes as arguments two custom structs with fields that
%   correspond to the point cloud attributes on which the structural
%   similarity will be computed. Note that "geom" field is mandatory in
%   order to permit neighborhood formulations and associations. The PARAMS
%   struct is used to configure the computation of structural similarity
%   scores.
%
%   Below you can find an example for the computation of structural
%   similarity scores using color-based features, with 'VAR' as the
%   dispersion estimator, 'Mean' as the pooling method, 12 points as the
%   neighborhood size, and using both point clouds as reference.


% clear all;
% close all;
% clc;


function [pcd_ssims, pcd_ids] = compute_ssim(ref_dir, dis_dir, n, total)
ssims = [];
ids = [];
for i = 0:total
    % pcd_dis_name = sprintf('./dis_noadapt/node0_%d.pcd', i);
    % pcd_dis_name = sprintf('%s/node0_frame0_chunk0.pcd', dir);
    pcd_dis_name = sprintf('%s/node0_frame%d.pcd', dis_dir, i);
    if isfile(pcd_dis_name)
        pcd_ref_name = sprintf('%s/%06d_%d.pcd', ref_dir, mod(i, 80), n);
        %% Load point clouds
        a = pcread(pcd_ref_name);
        b = pcread(pcd_dis_name);
        %% Define structs with required fields
        pcA.geom = a.Location;
        pcA.color = a.Color;
        pcB.geom = b.Location;
        pcB.color = b.Color;
        %% Configure PARAMS
        PARAMS.ATTRIBUTES.GEOM = true;
        PARAMS.ATTRIBUTES.NORM = false;
        PARAMS.ATTRIBUTES.CURV = false;
        PARAMS.ATTRIBUTES.COLOR = false;
        PARAMS.ESTIMATOR_TYPE = {'VAR'};
        PARAMS.POOLING_TYPE = {'Mean'};
        PARAMS.NEIGHBORHOOD_SIZE = 12;
        PARAMS.CONST = eps(1);
        PARAMS.REF = 0;
        %% Compute structural similarity values based on selected PARAMS
        [pointssim] = pc_ssim(pcA, pcB, PARAMS);
        % disp(pointssim)
        % disp(pointssim.geomBA);
        ids(end + 1) = i;
        ssims(end + 1) = pointssim.geomSym;
    end
end
% for i = 1:size(ssims, 2)
%     % disp(ssims(i));
%     fprintf('%d %f\n', ids(i), ssims(i));
% end
% disp(ssims);
% disp(size(ssims))
% disp(mean(ssims));
% disp(std(ssims));
pcd_ssims = ssims 
pcd_ids = ids
% end



% %% Load point clouds
% a = pcread('000000.pcd');
% b = pcread('000000.pcd');


% %% Define structs with required fields
% pcA.geom = a.Location;
% pcA.color = a.Color;

% pcB.geom = b.Location;
% pcB.color = b.Color;


% %% Configure PARAMS
% PARAMS.ATTRIBUTES.GEOM = true;
% PARAMS.ATTRIBUTES.NORM = false;
% PARAMS.ATTRIBUTES.CURV = false;
% PARAMS.ATTRIBUTES.COLOR = false;

% PARAMS.ESTIMATOR_TYPE = {'VAR'};
% PARAMS.POOLING_TYPE = {'Mean'};
% PARAMS.NEIGHBORHOOD_SIZE = 12;
% PARAMS.CONST = eps(1);
% PARAMS.REF = 0;


% %% Compute structural similarity values based on selected PARAMS
% [pointssim] = pc_ssim(pcA, pcB, PARAMS);
% disp(pointssim)
