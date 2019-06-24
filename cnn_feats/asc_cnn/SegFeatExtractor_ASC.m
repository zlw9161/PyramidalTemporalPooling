function SegFeatExtractor_ASC

featdir = 'D:\Experiments\cnn_feats\asc_cnn\feats\ldrbrd_logmel120.mat';
labeldir = 'D:\Experiments\cnn_feats\asc_cnn\feats\ldrbrd_logmel120_label.mat';
datalistdir = 'D:\Experiments\data\dcase2018\lists\ldrbrd_audio.txt';
labellistdir = 'D:\Experiments\data\dcase2018\lists\ldrbrd_label.txt';
datalist = importdata(datalistdir);
labellist = importdata(labellistdir);
datanum = size(datalist,1);
label = zeros(datanum,1);
disp(datanum);
seglen = 48960; % 48960 for 1sec patch
seg_shift = 48000*0.75; % 24000 for 1 sec
isovlp = 1;
window = 1920; % 960 for 40 ms / 1 sec
frm_shift = 960; % 240 for mfcc
patch_size = floor((seglen - (window - frm_shift)) / frm_shift);
feat_dim = 120;% mfcc:126; spctgrm: nfft/2+1; logmel: (PdD) mel_bank num * 3;
[segnum, totalsegs] = cal_segnum(datalist,seglen,seg_shift,isovlp);
if isovlp == 1
    feats = zeros(patch_size, feat_dim, totalsegs, 'single');
else
    feats = zeros(patch_size, feat_dim, totalsegs, 'single');
    seg_shift = seglen;
end
k = 0;
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
    for j = 1 : segnum(i)
        if isovlp == 0
            head = 1 + (j-1) * seglen; 
            tail = j * seglen;
            %tmp = melcepst(x(head:tail), Fs, 'E0dD', 40, 128, window, frm_shift, 0, 0.5);
            %tmp = spectrogram(x(head:tail), window, frm_shift, 512, Fs)';
            tmp = logmelbanks(x(head:tail), Fs, 'p0', feat_dim, window, frm_shift, 0, 0.5);
            % for log mel feats
            tmp(:,end) = [];
            %tmp = zscore(tmp);
            %disp(size(tmp));
            feats(1:size(tmp,1), :, k+j) = single(tmp);
        else
            head = 1 + (j-1) * seg_shift;
            tail = (j-1) * seg_shift + seglen;
            %tmp = melcepst(x(head:tail), Fs, 'E0dD', 40, 128, window, frm_shift, 0, 0.5);
            %tmp = spectrogram(x(head:tail), window, frm_shift, 512, Fs)';
            tmp = logmelbanks(x(head:tail), Fs, 'p0', feat_dim, window, frm_shift, 0, 0.5);
            % for log mel feats
            tmp(:,end) = [];
            %tmp = zscore(tmp);
            %disp(size(tmp));
            feats(1:size(tmp,1), :, k+j) = single(tmp);
        end
        scene = labellist{i};
        switch scene 
            case 'airport'
                label(k+j) = 1;
            case 'bus'
                label(k+j) = 2;
            case 'metro'
                label(k+j) = 3;
            case 'metro_station'
                label(k+j) = 4;
            case 'park'
                label(k+j) = 5;
            case 'public_square'
                label(k+j) = 6;
            case 'shopping_mall'
                label(k+j) = 7;
            case 'street_pedestrian'
                label(k+j) = 8;
            case 'street_traffic'
                label(k+j) = 9;
            case 'tram'
                label(k+j) = 10;
        end
    end
    k = k + segnum(i);
    disp(i);
end
disp(totalsegs);
% disp(k);
labels = label';
save(labeldir,'labels');
save(featdir,'feats');
fprintf('Complete...\n');
end

% -------------------------------------------------------------------------
function [segnum, totalsegs] = cal_segnum(datalist,seglen,shift,isovlp)
% -------------------------------------------------------------------------
datanum = size(datalist, 1);
segnum = zeros(datanum, 1);

% for k = 1 : datanum
%     %disp(k);
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