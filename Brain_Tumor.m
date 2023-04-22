%Brain Tumor Classification
% Loading brain tumor image dataset
imds = imageDatastore("C:\Users\DELL\Downloads\brain_tumor binary", 'IncludeSubfolders',true, 'LabelSource','foldernames');
inputSize = [26 26];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);   %Bringing all the input images to 26*26

labelCount = countEachLabel(imds);
perm = randperm(500,12);       

%Visualising some images
figure;
for i = 1:12
    subplot(3,4,i);
    imshow(imds.Files{perm(i)});
end


numTrainFiles = 0.80;
[imdsTrain,imdsValidation] = splitEachLabel(imds,numTrainFiles,'randomize');           

%Splitting each label in a 8:1 ratio
YValidation = imdsValidation.Labels;
inputSize=[26 26 1];
imdsTrain=augmentedImageDatastore(inputSize, imdsTrain,'ColorPreprocessing','rgb2gray');
imdsValidation=augmentedImageDatastore(inputSize, imdsValidation,'ColorPreprocessing','rgb2gray');

%Defining CNN architechture
layers = [
    imageInputLayer([26 26 1])


            convolution2dLayer(3,8,'Padding','same','Name','Conv_1')
            batchNormalizationLayer('Name','BN_1')
            reluLayer('Name','Relu_1')
            maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_1')
            
                      
            
            convolution2dLayer(3,16,'Padding','same','Name','Conv_2')
            batchNormalizationLayer('Name','BN_2')
            reluLayer('Name','Relu_2')
            maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_2')
            
                      
            
            convolution2dLayer(3,32,'Padding','same','Name','Conv_3')
            batchNormalizationLayer('Name','BN_3')
            reluLayer('Name','Relu_3')
            maxPooling2dLayer(2,'Stride',2,'Name','MaxPool_3')

      
            
            convolution2dLayer(3,64,'Padding','same','Name','Conv_4')
            batchNormalizationLayer('Name','BN_4')
            reluLayer('Name','Relu_4')

            convolution2dLayer(3,64,'Padding','same','Name','Conv_5')
            batchNormalizationLayer('Name','BN_5')
            reluLayer('Name','Relu_5')
 
            fullyConnectedLayer(2,'Name','FC')
            softmaxLayer('Name','SoftMax');
            classificationLayer('Name','Output Classification');

    ];

lgraph = layerGraph(layers);
figure;
plot(lgraph);

%Adding the training options

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrain,layers,options);
YPred = classify(net,imdsValidation);


% Evaluate the performance metrics of the model
accuracy= mean(YPred == YValidation)
confusion_matrix = confusionmat(YValidation, YPred)
precision = confusion_matrix(2,2) / sum(confusion_matrix(:,2))
recall = confusion_matrix(2,2) / sum(confusion_matrix(2,:))
f1_score = 2 * precision * recall / (precision + recall)


plotconfusion(YValidation,YPred)