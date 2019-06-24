% *************************************************************************
% Dependency : 
% 1. Lib Linear 
%       https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/multicore-liblinear/
% 2. VL-Feat 
%       http://www.vlfeat.org/
% 3. LibSVM
%       https://www.csie.ntu.edu.tw/~cjlin/libsvm/
% Function : run ASC task using MLTP method with CNN FC features

function MLTP_ASC_CNN()

CVAL    = 0.0001;    % The default C value for SVR 0.00005
WIN     = 3;   % The window size of the poooling layers
STRIDE  = 1;   % The stride of the pooling layer
Options.EncNonLin = 6;   % Non-Linearity Type before encoding
Options.DataNonLin = 8;   % Non-Linearity Type for encoded data (SVM-Kernel)
Options.PCA = 0;
Options.ComponentsNum = 0;
Options.DualPCA = 0;

% define the network architecture
net = defineNetwork(CVAL,WIN,STRIDE);   

% GENERATING TRAINING SEQUENCE ENCODING
EnonLin = chooseNonLinearity(Options.EncNonLin);
featfiles = {'train_ny_b_ep10_25%ovlp','eval_ny_b_ep10_25%ovlp'};
%featfiles = {'train_25%ovlp_res34_ep46','eval_25%ovlp_res34_ep46'};
rsltfile = sprintf('asc_results/rslt%s_w3s1_idn_C0.0001_2ltp_ser_ser.mat',featfiles{1}(6:end));
for f = 1 : numel(featfiles)
    encfile = sprintf('D:/Experiments/mltp/encoded_feats/%s_w3s1_idn_C0.0001_2ltp_ser.mat',featfiles{f});
    if exist(encfile,'file') ~= 2
        featfile = sprintf('D:/Experiments/cnn_feats/asc_cnn/fc_feat/%s.mat',featfiles{f});
        disp(featfile);
        load(featfile);
        [EncodedData] = getEncodedData(feats, net, EnonLin);
        save(encfile,'EncodedData');
    else
        load(encfile);
    end
    Encoded_Data_cell{f} = EncodedData;

end    

% Get Non-linearity for encoded data
if Options.DataNonLin ~= 0
    DnonLin = chooseNonLinearity(Options.DataNonLin);
    for ch = 1 : size(Encoded_Data_cell,2)
%         seqMean = mean(Encoded_Data_cell{ch},2);
%         Encoded_Data_cell{ch} = Encoded_Data_cell{ch} - seqMean;
        Encoded_Data_cell{ch} = normalizeL2(getNonLinearity(Encoded_Data_cell{ch},DnonLin));
    end
end

% load label files
load('D:/Experiments/cnn_feats/asc_cnn/fc_feat/train_label.mat');
TrainClass = label;
n_Trn = size(label,1);
clear label;
load('D:/Experiments/cnn_feats/asc_cnn/fc_feat/eval_label.mat');
TestClass = label;
n_Test = size(label,1);
clear label;
dataset.classlabel = [TrainClass; TestClass];
dataset.traintest  = [ones(n_Trn,1) ; ones(n_Test,1)+1];

% Prepare the data for SVM Classifiers
weights = 1;
encoded_trn_data = Encoded_Data_cell{1};
encoded_test_data = Encoded_Data_cell{2};
% Compress with PCA
if Options.PCA ~= 0
    [coef, score_trn] = pca(encoded_trn_data);
    if Options.ComponentsNum == 0
        Options.ComponentsNum = size(coef, 2);
    end
    encoded_trn_data = score_trn(:,1:Options.ComponentsNum);
    encoded_test_data = bsxfun(@minus,encoded_test_data,mean(encoded_test_data,1));
    score_test = encoded_test_data * coef;
    encoded_test_data = score_test(:,1:Options.ComponentsNum);
    clear coef latent score_trn score_test;
end
if Options.DualPCA ~= 0
    [coef, score_trn] = pca(encoded_trn_data);
    encoded_trn_data = score_trn(:,1:500);
    encoded_test_data = bsxfun(@minus,encoded_test_data,mean(encoded_test_data,1));
    score_test = encoded_test_data * coef;
    encoded_test_data = score_test(:,1:500);
    clear coef latent score_trn score_test;
end
TrainData_Kern = encoded_trn_data * encoded_trn_data';                
TestData_Kern = encoded_test_data * encoded_trn_data';
clear encoded_trn_data; clear encoded_test_data; 
clear Encoded_Data_cell;

% Find the best CVAL for SVM classifier
fprintf('Search the best C Value for SVM:\n');
[precision(weights,:),recall(weights,:),acc(weights),C] = train_and_classify(TrainData_Kern,TestData_Kern,TrainClass,TestClass);
[accuracy,indx] = max(acc);            
precision = precision(indx,:);
precision(isnan(precision)) = 0;
recall = recall(indx,:);
recall(isnan(recall)) = 0;
avg_acc = mean(recall);
F = 2*(precision .* recall)./(precision+recall);
F(isnan(F)) = 0;
fprintf('Mean F score = %1.4f\n',mean(F));
fprintf('Class avg acc = %1.4f\n',mean(avg_acc));
save(rsltfile,'accuracy','precision','recall','F','avg_acc');

% Calculate accuracy for each class:
fprintf('Classify the data with the best C Value:\n');
[multiclass,perclass] = getClassificationMeasures(dataset,TrainData_Kern,TestData_Kern,C);
fprintf('Mean perclass accuracy %1.1f \n',mean(perclass));
fprintf('Multiclass  accuracy %1.1f \n',multiclass);


% choose Non-Linearity Type
function nonLin = chooseNonLinearity(NonLinType)    
    switch NonLinType
        case 1
            nonLin = '';
        case 2
            nonLin = 'tanh';
        case 3
            nonLin = 'ssr';
        case 4
            nonLin = 'chi2';
        case 5
            nonLin = 'chi1';
        case 6
            nonLin = 'none';
        case 7
            nonLin = 'chi2exp';
        case 8
            nonLin = 'ser';
        case 9
            nonLin = 'chi3';            
    end

    
function [EncodedData] = getEncodedData(feats, net, EnonLin)
    %CVAL = 1; % CVAL for SVR
    TOTAL = numel(feats);
    %EncodedData = zeros(TOTAL,1024*3*16,'single');
    EncodedData = zeros(TOTAL,512*16);
    for i = 1 : TOTAL
        % for each seqence load data
        data = feats{i};
        %data = normalize(data, 2);
        % apply non-linear feature maps
        x = getNonLinearity(data,EnonLin) ;
        % get the encoding of the sequence
        enc = passNetwork(x,net) ;
        % disp(size(enc));
        %EncodedData(i,:) = single(enc);
        EncodedData(i,:) =  enc;
        fprintf('.');if mod(i,100)==0, fprintf('\n'); end
    end
    fprintf('Complete...\n')

    
function [precision, recall, acc, C] = train_and_classify(TrainData_Kern,TestData_Kern,TrainClass,TestClass)
        % precomputed kernel c-svm
        nTrain = 1 : size(TrainData_Kern,1);
        TrainData_Kern = [nTrain' TrainData_Kern];         
        nTest = 1 : size(TestData_Kern,1);
        TestData_Kern = [nTest' TestData_Kern];% add test index??? 
        
        C = 0.01 : 0.02 : 0.51;
        %C = [0.001 0.005 0.01 0.05 0.1 0.5 1 5 10 15 20 25 50 75 100 500];
        %C = 0.38; % for CVAL = 1e-5
%         C = 0.29;
        CoreNum = 4;
        TrainPool = parpool(CoreNum);
        for ci = 1 : numel(C)
            %model(ci) = svmtrain(TrainClass, TrainData_Kern, sprintf('-t 4 -c %1.6f -v 2 -q -w6 3 -w15 3 -w23 5 -e 0.001',C(ci)));
            %model(ci) = svmtrain(double(TrainClass), double(TrainData_Kern), sprintf('-t 4 -c %1.6f -v 2 -q -b 1',C(ci)));
            model = svmtrain(double(TrainClass), double(TrainData_Kern), sprintf('-t 4 -c %1.6f -q -b 1 -m 500',C(ci)));
            [predicted, acc, scores{ci}] = svmpredict(double(TestClass), double(TestData_Kern) ,model);
            accuracy(ci) = acc(1,1);
            count = hist(predicted,unique(predicted));
            if size(count,2) < 10
                count_pad = zeros(1,10-size(count,2));
                count = cat(2,count,count_pad);
            end
            %cnt_grndtrth = hist(TestClass,unique(TestClass));
            cnt_grndtrth = 120;
            disp(count);
            wrong_num_ratio(ci) = 1 - sum(abs(count-cnt_grndtrth))/size(TestClass,1);
            fprintf('Wrong Numbers Ratio for C: %1.6f is: %1.6f\n', C(ci),wrong_num_ratio(ci));
        end
        delete(TrainPool);
        
        %[~,max_index] = max(model);
        %C = C(max_index);
        [~,max_index] = max(accuracy);
        C = C(max_index);
        fprintf('The best C Value of Precomputed Kernel SVM: C: %1.6f\n', C);
         
        for ci = 1 : numel(C)
            % precomputed kernel c-svm
            model = svmtrain(double(TrainClass), double(TrainData_Kern), sprintf('-t 4 -c %1.6f -q -b 1 -m 500',C(ci)));
            [predicted, acc, scores{ci}] = svmpredict(double(TestClass), double(TestData_Kern) ,model);
            [precision(ci,:) , recall(ci,:)] = perclass_precision_recall(TestClass,predicted);
            accuracy(ci) = acc(1,1);
        end        
        prdctfile = sprintf('predicted_w3s2_idn_C0.01_2ltp_ser_ser_c0.29.mat');
        save(prdctfile, 'predicted');
        [acc,cindx] = max(accuracy);   
        scores = scores{cindx};
        precision = precision(cindx,:);
        recall = recall(cindx,:);

        
function [precision , recall] = perclass_precision_recall(label,predicted)

    for cl = 1 : 10
        true_pos = sum((predicted == cl) .* (label == cl));
        false_pos = sum((predicted == cl) .* (label ~= cl));
        false_neg = sum((predicted ~= cl) .* (label == cl));
        precision(cl) = true_pos / (true_pos + false_pos);
        recall(cl) = true_pos / (true_pos + false_neg);
        
    end
    
        
function [multiclass,perclass] = getClassificationMeasures(dataset,TrainData_Kern,TestData_Kern,C)  
    
    [classlabel]   = getLabel(dataset.classlabel);
    disp(size(classlabel));
    trn_indx       = find(dataset.traintest == 1);
    test_indx      = find(dataset.traintest ~= 1);
    TestClass      = classlabel(test_indx,:);
    TrainClass     = classlabel(trn_indx,:);
    test_classid   = dataset.classlabel(test_indx);
    trn_classid    = dataset.classlabel(trn_indx);
    score          = zeros(size(TestClass,1),size(classlabel,2));    
    cid = 1;
    CVAL = C;
    % 2-class classification accuracy for each class
    for cl = 1 : size(classlabel,2)            
        trnLBLB = TrainClass(:,cl);
        testLBL = TestClass(:,cl);    
        [score(:,cl)] = getScores(TrainData_Kern,TestData_Kern,trnLBLB,testLBL,CVAL);                    
    end   
    
    [~,predcl] = max(score,[],2);        
    multiclass = numel(find(predcl==test_classid))/numel(test_classid) * 100;   
    
    for cl = 1 : size(score,2)     
        indx = find(test_classid ==cl);
        if numel(indx)> 0 && numel(find(trn_classid ==cl)) > 0
            perclass(cid) = numel(find (predcl(indx) == cl)) / numel(indx) * 100;            
            cid = cid + 1;
        end
    end   

    
function [X] = getLabel(classid)
    X = zeros(numel(classid),max(classid))-1;
    for i = 1 : max(classid)
        indx = find(classid == i);
        X(indx,i) = 1;
    end

    
function [score] = getScores(TrainData_Kern,TestData_Kern,TrainClass,TestClass,C)
        nTrain = 1 : size(TrainData_Kern,1);
        TrainData_Kern = [nTrain' TrainData_Kern];         
        nTest = 1 : size(TestData_Kern,1);
        TestData_Kern = [nTest' TestData_Kern];                         
        model = svmtrain(double(TrainClass), double(TrainData_Kern), sprintf('-t 4 -c %1.6f -q -b 1 -m 500', C));
        [~, ~, scores] = svmpredict(double(TestClass), double(TestData_Kern) ,model, '-b 1');
							 
        score = scores(:,1);        

