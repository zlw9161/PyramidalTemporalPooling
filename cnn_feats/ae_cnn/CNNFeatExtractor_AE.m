function CNNFeatExtractor_AE

featdir = '/data/Leon/Matconvnet/ae_cnn/fc_feat/test_fc.mat';
labeldir = '/data/Leon/Matconvnet/ae_cnn/fc_feat/test_label.mat';
datalistdir = '/data/Leon/Data/AudioEventDataset/list/test_audio.txt';
labellistdir = '/data/Leon/Data/AudioEventDataset/list/test_label.txt';
aedbdir = '/data/Leon/Matconvnet/ae_cnn/spct_a_bnorm/aedb.mat';
netdir = '/data/Leon/Matconvnet/ae_cnn/spct_a_bnorm/net_params.mat';
datalist = importdata(datalistdir);
labellist = importdata(labellistdir);

% get dataMean and dataStd
aedb = load(aedbdir);
dataMean = aedb.mfccs.data_mean;
dataStd = aedb.mfccs.data_std;
clear aedb;

% load pre-tested net model
pretrndnet = load(netdir);

% Get FC Feats
datanum = size(datalist,1);
label = zeros(datanum,1);
disp([num2str(datanum),' audio files in total...']);
seglen = 16480;
seg_shift = 16000*0.5;
isovlp = 1;
window = 640;
frm_shift = 160;
patch_size = floor((seglen - (window - frm_shift)) / frm_shift);
input_dim = 129;% mfcc:150; spctgrm: nfft/2+1; logmel: 90;
segnum = cal_segnum(datalist,seglen,seg_shift,isovlp);

for i = 1 : datanum
    [x,Fs] = audioread(datalist{i});
    if size(x,1) < (seglen * 3)
        x = cat(1, x, x);
    else
        x = cat(1, x, x((size(x,1)-seglen+1):end,:));
    end
    ifc = zeros(segnum(i), 1024);
    for j = 1 : segnum(i)
        if isovlp == 0
            head = 1 + (j-1) * seglen; 
            tail = j * seglen;
            tmp = spectrogram(x(head:tail), window, window-frm_shift, 256, Fs)';
            input_feat = single(reshape(tmp, patch_size, input_dim, 1));
            input_feat = bsxfun(@minus, input_feat, dataMean);
            input_feat = bsxfun(@rdivide, input_feat, dataStd);
            output_fc = getCNNFC(pretrndnet, input_feat);
            ifc(j, :) = output_fc;
            if j == segnum(i)
                feats{i} = ifc;
                clear ifc;
            end
        else
            head = 1 + (j-1) * seg_shift;
            tail = (j-1) * seg_shift + seglen;
            tmp = spectrogram(x(head:tail), window, window-frm_shift, 256, Fs)';
            input_feat = single(reshape(tmp, patch_size, input_dim, 1));
            input_feat = bsxfun(@minus, input_feat, dataMean);
            input_feat = bsxfun(@rdivide, input_feat, dataStd);
            output_fc = getCNNFC(pretrndnet, input_feat);
            ifc(j, :) = output_fc;
            if j == segnum(i)
                feats{i} = ifc;
                clear ifc;
            end
        end
    end
    disp(['#',num2str(i),' audio file CNNFeats complete...']);
end
save(featdir,'feats');
fprintf('Getting CNNFeats Complete...\n');

% Get Labels
for i = 1 : datanum
    category = labellist{i};
    switch category 
        case 'acoustic_guitar'
            label(i) = 1;
        case 'airplane'
            label(i) = 2;
        case 'applause'
            label(i) = 3;
        case 'bird'
            label(i) = 4;
        case 'car'
            label(i) = 5;
        case 'cat'
            label(i) = 6;
        case 'child'
            label(i) = 7;
        case 'church_bell'
            label(i) = 8;
        case 'crowd'
            label(i) = 9;
        case 'dog_barking'
            label(i) = 10;
        case 'engine'
            label(i) = 11;
        case 'fireworks'
            label(i) = 12;
        case 'footstep'
            label(i) = 13;
        case 'glass_breaking'
            label(i) = 14;
        case 'hammer'
            label(i) = 15;
        case 'helicopter'
            label(i) = 16;
        case 'knock'
            label(i) = 17;
        case 'laughter'
            label(i) = 18;
        case 'mouse_click'
            label(i) = 19;
        case 'ocean_surf'
            label(i) = 20;
        case 'rustle'
            label(i) = 21;
        case 'scream'
            label(i) = 22;
        case 'speech_fs'
            label(i) = 23;
        case 'squeak'
            label(i) = 24;
        case 'tone'
            label(i) = 25;
        case 'violin'
            label(i) = 26;
        case 'water_tap'
            label(i) = 27;
        case 'whistle'
            label(i) = 28;
    end         
end
%labels = label';
save(labeldir,'label');
fprintf('Getting Labels Complete...\n');

end

% -------------------------------------------------------------------------
function [segnum, totalsegs] = cal_segnum(datalist,seglen,shift,isovlp)
% -------------------------------------------------------------------------
datanum = size(datalist, 1);
segnum = zeros(datanum, 1);
for k = 1 : datanum
    [x, ~] = audioread(datalist{k});
    if size(x,1) < (seglen * 3)
        x = cat(1, x, x);
    else
        x = cat(1, x, x((size(x,1)-seglen+1):end,:));
    end
    audiolen = size(x,1);
    if isovlp == 0
        segnum(k) = floor(audiolen / seglen);
    else
        if (audiolen-(floor(audiolen/shift)-1)*shift) < (seglen-shift) 
            segnum(k) = floor(audiolen / shift) - 2;
        else
            segnum(k) = floor(audiolen / shift) - 1;
        end
    end 
end        
totalsegs = sum(segnum);
end

% -------------------------------------------------------------------------
function fc = getCNNFC(model, input)
% -------------------------------------------------------------------------
model.net.layers{end}.class = 1 ; % only get the fc layer output, the label doesn't need to be true
res = [] ;
dzdy = [] ;
Mode = 'test' ;
s = 1 ;
res = vl_simplenn(model.net, input, dzdy, res, ...
                  'accumulate', s ~= 1, ...
                  'mode', Mode) ;
fc = reshape(res(end-4).x, 1, 1024); % fc before activition
end