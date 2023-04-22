[filename, pathname] = uigetfile('*.*','Select the Input Image:');
filewithpath = strcat(pathname,filename);
I = imread(filewithpath);
I = imresize(I,[26 26]);
[rows,cols,chns,~] = size(I);
if chns==3
    I= rgb2gray(I);
end    

figure;
imshow(I);
label = classify(net,I);
title(['Brain Tumor Found: ' ...
,char(label)])
