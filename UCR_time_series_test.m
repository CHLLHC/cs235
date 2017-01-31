function UCR_time_series_test %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% (C) Eamonn Keogh %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
TRAIN = load('dota2Train.csv'); % Only these two lines need to be changed to test a different dataset  %
TEST  = load('dota2Test.csv' ); % Only these two lines need to be changed to test a different dataset  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


TRAIN_class_labels = TRAIN(:,1);     % Pull out the class labels.
TRAIN(:,1) = [];                     % Remove class labels from training set.
%TRAIN(:,1) = [];                     % Remove number.
[Class,Count] = mode(TRAIN_class_labels);
Default = 1-(Count/size(TRAIN_class_labels,1));
TEST_class_labels = TEST(:,1);       % Pull out the class labels.
TEST(:,1) = [];                      % Remove class labels from testing set.

Olcorrect = LeaveOne(TRAIN, TRAIN_class_labels); % Initialize the number we got correct
Ohcorrect = HoldOut(TRAIN, TRAIN_class_labels, TEST, TEST_class_labels);

ZOneTRAIN = bsxfun(@minus,TRAIN, min(TRAIN));
ZOneTRAIN = bsxfun(@rdivide,ZOneTRAIN, max(ZOneTRAIN));
ZOneTEST = bsxfun(@minus,TEST, min(TEST));
ZOneTEST = bsxfun(@rdivide,ZOneTEST, max(ZOneTEST));
Zolcorrect = LeaveOne(ZOneTRAIN, TRAIN_class_labels);
Zohcorrect = HoldOut(ZOneTRAIN, TRAIN_class_labels, ZOneTEST, TEST_class_labels);

ZTRAIN = zscore(TRAIN);
ZTEST = zscore(TEST);
Zlcorrect = LeaveOne(ZTRAIN, TRAIN_class_labels);
Zhcorrect = HoldOut(ZTRAIN, TRAIN_class_labels, ZTEST, TEST_class_labels);


%%%%%%%%%%%%%%%%% Create Report %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['The dataset you tested has ', int2str(length(unique(TRAIN_class_labels))), ' classes'])
disp(['The training set is of size ', int2str(size(TRAIN,1)),', and the test set is of size ',int2str(size(TEST,1)),'.'])
disp(['The time series are of length ', int2str(size(TRAIN,2))])
disp(['The default rate was ', num2str(Default)])
disp(['The Original LeaveOne error rate was ',num2str((length(TRAIN_class_labels)-Olcorrect )/length(TRAIN_class_labels))])
disp(['The Z-one LeaveOne error rate was ',num2str((length(TRAIN_class_labels)-Zolcorrect )/length(TRAIN_class_labels))])
disp(['The Z LeaveOne error rate was ',num2str((length(TRAIN_class_labels)-Zlcorrect )/length(TRAIN_class_labels))])
disp(['The Original Hold-out error rate was ',num2str((length(TEST_class_labels)-Ohcorrect )/length(TEST_class_labels))])
disp(['The Z-one Hold-out error rate was ',num2str((length(TEST_class_labels)-Zohcorrect )/length(TEST_class_labels))])
disp(['The Z Hold-out error rate was ',num2str((length(TEST_class_labels)-Zhcorrect )/length(TEST_class_labels))])
%%%%%%%%%%%%%%%%% End Report %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end

function ans = LeaveOne(TRAIN, TRAIN_class_labels)
    correct = 0;
    for i = 1 : length(TRAIN_class_labels) % Loop over every instance in the test set
       classify_this_object = TRAIN(i,:);
       this_objects_actual_class = TRAIN_class_labels(i);
       predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels, classify_this_object, i);
       if predicted_class == this_objects_actual_class
           correct = correct + 1;
       end;
       if (mod(i,100)==0)
        disp([int2str(i), ' out of ', int2str(length(TRAIN_class_labels)), ' done']); % Report progress
       end;
    end;
    ans = correct;
end


function ans = HoldOut(TRAIN, TRAIN_class_labels, TEST, TEST_class_labels)
    correct = 0;
    for i = 1 : length(TEST_class_labels) % Loop over every instance in the test set
       classify_this_object = TEST(i,:);
       this_objects_actual_class = TEST_class_labels(i);
       predicted_class = Classification_Algorithm(TRAIN,TRAIN_class_labels, classify_this_object, 0);
       if predicted_class == this_objects_actual_class
           correct = correct + 1;
       end;
       if (mod(i,100)==0)
        disp([int2str(i), ' out of ', int2str(length(TEST_class_labels)), ' done']); % Report progress
       end;
    end;
    ans = correct;
end

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
        distance = (sum((compare_to_this_object - unknown_object).^2)); % Euclidean distance
        if distance < best_so_far
          predicted_class = TRAIN_class_labels(i);
         best_so_far = distance;
        end
     end
 end;
end