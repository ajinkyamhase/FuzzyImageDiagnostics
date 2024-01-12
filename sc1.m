% Load the grayscale MRI image
MRI = imread('brain_mri.jpg');

% Convert the image to double data type for FCM
I = im2double(MRI);

% Define the number of clusters (2 for brain and tumor)
num_clusters = 2;

% Fuzzy C-Means clustering
[center, U, obj_fcn] = fcm(I, num_clusters);

% Threshold the membership values to create a binary image
threshold = 0.5; % You may need to adjust this threshold
brain_mask = U(1, :) >= threshold;
tumor_mask = U(2, :) >= threshold;

% Display the results
figure;
subplot(1, 3, 1);
imshow(I, []);
title('Original Image');

subplot(1, 3, 2);
imshow(brain_mask);
title('Brain Mask');

subplot(1, 3, 3);
imshow(tumor_mask);
title('Tumor Mask');
