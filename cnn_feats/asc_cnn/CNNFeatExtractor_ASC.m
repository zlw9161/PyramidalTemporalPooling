function CNNFeatExtractor_ASC

featdir = 'D:\Experiments\cnn_feats\asc_cnn\fc_feat\eval_fc.mat';
labeldir = 'D:\Experiments\cnn_feats\asc_cnn\fc_feat\eval_label.mat';
datalistdir = 'D:\Experiments\data\dcase2018\lists\eval_audio1.txt';
labellistdir = 'D:\Experiments\data\dcase2018\lists\eval_label1.txt';
aedbdir = 'D:\Experiments\cnn_feats\asc_cnn\ny_b_logmel120_bnorm\aedb_train.mat';
netdir = 'D:\Experiments\cnn_feats\asc_cnn\trained_models\net_params.mat';
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
seglen = 48960;
seg_shift = 48000*0.75;
isovlp = 1;
window = 1920;
frm_shift = 960;
patch_size = floor((seglen - (window - frm_shift)) / frm_shift);
input_dim = 120;% mfcc:150; spctgrm: nfft/2+1; logmel: 40;
segnum = cal_segnum(datalist,seglen,seg_shift,isovlp);
fcdim = 512;

for i = 1 : datanum
    [x,Fs] = audioread(datalist{i});
%     if size(x,1) < (seglen * 3)
%         x = cat(1, x, x);
%     else
%         x = cat(1, x, x((size(x,1)-seglen+1):end,:));
%     end
    audiolen = size(x,1);
    reslen = audiolen-floor(audiolen/seg_shift)*seg_shift;
    if reslen < seglen
        padlen = seglen-seg_shift-reslen;
        if padlen > 0
            x = cat(1,x,x((audiolen-padlen+1):end,:));
        else
            x = cat(1,x,x((audiolen+padlen+1):end,:));
        end
    end
    ifc = zeros(segnum(i), fcdim);
    for j = 1 : segnum(i)
        if isovlp == 0
            head = 1 + (j-1) * seglen; 
            tail = j * seglen;
            %tmp = spectrogram(x(head:tail), window, window-frm_shift, 256, Fs)';
            %tmp = melcepst(x(head:tail), Fs, 'E0dD', 40, 128, window, frm_shift, 0, 0.5);
            tmp = logmelbanks(x(head:tail), Fs, 'p0', input_dim, window, frm_shift, 0, 0.5);
            % for log mel feats
            tmp(:,end) = [];
            tmp = normalizeL2(tmp);
            input_feat = single(reshape(tmp, patch_size, input_dim, 1));
            input_feat = bsxfun(@minus, input_feat, dataMean);
            input_feat = bsxfun(@rdivide, input_feat, dataStd);
            output_fc = getCNNFC(pretrndnet, input_feat, fcdim);
            ifc(j, :) = output_fc;
            if j == segnum(i)
                feats{i} = ifc;
                clear ifc;
            end
        else
            head = 1 + (j-1) * seg_shift;
            tail = (j-1) * seg_shift + seglen;
            %tmp = spectrogram(x(head:tail), window, window-frm_shift, 256, Fs)';
            %tmp = melcepst(x(head:tail), Fs, 'E0dD', 40, 128, window, frm_shift, 0, 0.5);
            tmp = logmelbanks(x(head:tail), Fs, 'p0', input_dim, window, frm_shift, 0, 0.5);
            % for log mel feats
            tmp(:,end) = [];
            tmp = normalizeL2(tmp);
            input_feat = single(reshape(tmp, patch_size, input_dim, 1));
            input_feat = bsxfun(@minus, input_feat, dataMean);
            input_feat = bsxfun(@rdivide, input_feat, dataStd);
            output_fc = getCNNFC(pretrndnet, input_feat, fcdim);
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
        case 'airport'
            label(i) = 1;
        case 'bus'
            label(i) = 2;
        case 'metro'
            label(i) = 3;
        case 'metro_station'
            label(i) = 4;
        case 'park'
            label(i) = 5;
        case 'public_square'
            label(i) = 6;
        case 'shopping_mall'
            label(i) = 7;
        case 'street_pedestrian'
            label(i) = 8;
        case 'street_traffic'
            label(i) = 9;
        case 'tram'
            label(i) = 10;
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
% for k = 1 : datanum
%     [x, ~] = audioread(datalist{k});
%     if size(x,1) < (seglen * 3)
%         x = cat(1, x, x);
%     else
%         x = cat(1, x, x((size(x,1)-seglen+1):end,:));
%     end
%     audiolen = size(x,1);
%     if isovlp == 0
%         segnum(k) = floor(audiolen / seglen);
%     else
%         if (audiolen-(floor(audiolen/shift)-1)*shift) < (seglen-shift) 
%             segnum(k) = floor(audiolen / shift) - 2;
%         else
%             segnum(k) = floor(audiolen / shift) - 1;
%         end
%     end 
% end
if isovlp == 0
    shift = seglen;
end
for k = 1 : datanum
    %disp(k);
    [x, ~] = audioread(datalist{k});
    audiolen = size(x,1);
    reslen = audiolen-floor(audiolen/shift)*shift;
    if reslen < seglen
        padlen = seglen-shift-reslen;
        if padlen > 0
            x = cat(1,x,x((audiolen-padlen+1):end,:));
        else
            x = cat(1,x,x((audiolen+padlen+1):end,:));
        end
    end
    audiolen = size(x,1);
    segnum(k) = floor(audiolen / shift);
end
totalsegs = sum(segnum);
end

% --------------------------------------------------------------------
function data = normalizeL2(data)
% --------------------------------------------------------------------
for i = 1 : size(data, 4)
    if norm(data(:,:,:,i)) ~= 0
        data(:,:,:,i) = data(:,:,:,i)./norm(data(:,:,:,i)) ;
    end
end
end

% -------------------------------------------------------------------------
function fc = getCNNFC(model, input, fcdim)
% -------------------------------------------------------------------------
model.net.layers{end}.class = 1 ; % only get the fc layer output, the label doesn't need to be true
res = [] ;
dzdy = [] ;
Mode = 'test' ;
s = 1 ;
res = vl_simplenn(model.net, input, dzdy, res, ...
                  'accumulate', s ~= 1, ...
                  'mode', Mode) ;
fc = reshape(res(end-4).x, 1, fcdim); % fc before activition
%fc = reshape(res(end-3).x, 1, fcdim); % fc after activition
end