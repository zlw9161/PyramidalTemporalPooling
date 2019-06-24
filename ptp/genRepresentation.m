function W = genRepresentation(data,CVAL,nonLin)
    % calculate the time varying means for each frame
    OneToN = [1:size(data,1)]';    
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    % temporal pooling
    W_fow = liblinearsvr(getNonLinearity(Data,nonLin),CVAL,0); clear Data; 			
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    W_rev = liblinearsvr(getNonLinearity(Data,nonLin),CVAL,0); 			              
    W = [W_fow ; W_rev]; 
end

% function W = genRepresentation(data,CVAL,nonLin)
%     Data =  zeros(size(data,1)-1,size(data,2));
%     for j = 2 : size(data,1)                
%         Data(j-1,:) = mean(data(1:j,:));
%     end
%     Data = getNonLinearity(Data,nonLin);
%     %disp(size(Data));
%     W_fow = liblinearsvr(Data,CVAL,0); 			
%     order = 1:size(data,1);
%     [~,order] = sort(order,'descend');
%     data = data(order,:);
%     Data =  zeros(size(data,1)-1,size(data,2));
%     for j = 2 : size(data,1)                
%         Data(j-1,:) = mean(data(1:j,:));
%     end
%     Data = getNonLinearity(Data,nonLin);
%     W_rev = liblinearsvr(Data,CVAL,0); 			              
%     W = [W_fow ; W_rev];
%     %W = W_fow;
%     %disp(size(W));
% end