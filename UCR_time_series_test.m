function UCR_time_series_test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (C) Eamonn Keogh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TRAIN = load('leaf.csv'); % Only these two lines need to be changed to test a different dataset  %
TEST  = load('PhalangesOutlinesCorrect_TEST' ); % Only these two lines need to be changed to test a different dataset  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



TRAIN_class_labels = TRAIN(:,1);     % Pull out the class labels.
TRAIN(:,1) = [];                     % Remove class labels from training set.
TRAIN(:,1) = [];                     % Remove number.
TRAIN = bsxfun(@minus,TRAIN, min(TRAIN));
TRAIN = bsxfun(@rdivide,TRAIN, max(TRAIN));
[Class,Count] = mode(TRAIN_class_labels);
Default = 1-(Count/size(TRAIN_class_labels,1));
TEST_class_labels = TEST(:,1);       % Pull out the class labels.
TEST(:,1) = [];                      % Remove class labels from testing set.
correct = 0; % Initialize the number we got correct



for i = 1 : length(TRAIN_class_labels) % Loop over every instance in the test set
   classify_this_object = TRAIN(i,:);
   this_objects_actual_class = TRAIN_class_labels(i);
   %MinusThis = TRAIN;
   %MinusThis(i,:) = [];
   %MinusThis_class_labels = TRAIN_class_labels;
   %MinusThis_class_labels(i,:) = [];
   %predicted_class = Classification_Algorithm(MinusThis,MinusThis_class_labels, classify_this_object);
   predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels, classify_this_object, i);
   if predicted_class == this_objects_actual_class
       correct = correct + 1;
   end;
   if (mod(i,10)==0)
    disp([int2str(i), ' out of ', int2str(length(TRAIN_class_labels)), ' done']) % Report progress
   end;
end;

%%%%%%%%%%%%%%%%% Create Report %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['The dataset you tested has ', int2str(length(unique(TRAIN_class_labels))), ' classes'])
disp(['The training set is of size ', int2str(size(TRAIN,1)),', and the test set is of size ',int2str(size(TEST,1)),'.'])
disp(['The time series are of length ', int2str(size(TRAIN,2))])
disp(['The default rate was ', num2str(Default)])
disp(['The error rate was ',num2str((length(TRAIN_class_labels)-correct )/length(TRAIN_class_labels))])
%%%%%%%%%%%%%%%%% End Report %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Here is a sample classification algorithm, it is the simple (yet very competitive) one-nearest
% neighbor using the Euclidean distance.
% If you are advocating a new distance measure you just need to change the line marked "Euclidean distance"
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels,unknown_object, excludeIndex)
best_so_far = inf;
 
 for i = 1 : length(TRAIN_class_labels)
     if (i ~= excludeIndex)
        compare_to_this_object = TRAIN(i,:);            
        distance = sqrt(sum((compare_to_this_object - unknown_object).^2)); % Euclidean distance
        if distance < best_so_far
          predicted_class = TRAIN_class_labels(i);
         best_so_far = distance;
        end
     end
end;