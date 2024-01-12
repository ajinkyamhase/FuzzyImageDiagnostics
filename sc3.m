% Prompt the user to select an image file
[filename, pathname] = uigetfile({'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff', 'Image Files (*.jpg, *.jpeg, *.png, *.bmp, *.tif, *.tiff)'}, 'Select an Image');
if isequal(filename, 0)
    disp('No image selected. Exiting.');
    return;
end

% Construct the full path to the selected image
image_path = fullfile(pathname, filename);

% Load the image
I = imread(image_path);
I = im2double(I); % Convert to double data type

% Define the number of clusters (2 for brain and tumor)
num_clusters = 2;

% Fuzzy C-Means clustering
[center, U, obj_fcn] = fcm(I(:), num_clusters);

% Determine the cluster for each pixel
[~, idx] = max(U);

% Reshape the segmented image to the original dimensions
segmented_image = reshape(idx, size(I));

% Display the original and segmented images
figure;
subplot(1, 2, 1);
imshow(I, []);
title('Original Image');

subplot(1, 2, 2);
imshow(segmented_image, []);
title('Segmented Image');

% Optional: Apply post-processing steps if needed
% E.g., morphological operations, noise removal, etc.

% Optional: Save the segmented image
% imwrite(segmented_image, 'segmented_image.jpg');

% Optional: Calculate statistics or region properties for further analysis
% ... (Previous code for loading and segmentation)

% Create a mask for the abnormal region (tumor)
threshold = 0.5; % You may need to adjust this threshold
abnormal_mask = segmented_image == 2; % Assuming tumor is cluster 2

% Optionally, apply morphological operations to further refine the abnormal region
se = strel('disk', 5); % Define a disk-shaped structuring element
abnormal_mask = imopen(abnormal_mask, se); % Perform morphological opening

% Overlay the abnormal region on the original image
rgb_image = cat(3, I, I, I); % Convert to an RGB image
rgb_image(repmat(abnormal_mask, [1, 1, 3])) = 1; % Overlay in red

% Display the original image with the abnormalities highlighted
figure;
subplot(1, 2, 1);
imshow(I, []);
title('Original Image');
% 
subplot(1, 2, 2);
imshow(rgb_image, []);
title('Abnormalities Highlighted');

% Optional: Save the segmented image or the overlay image
% imwrite(segmented_image, 'segmented_image.jpg');
% imwrite(rgb_image, 'abnormalities_highlighted.jpg');

% Optional: Calculate statistics or region properties for further analysis
