function x = normalizeL2(x)
    for i = 1 : size(x,1)
		if norm(x(i,:)) ~= 0
			x(i,:) = x(i,:) ./ norm(x(i,:));
		end
    end
%     v = sqrt(sum(x.*conj(x),2));
%     v(find(v==0))=1;
%     x=x./repmat(v,1,size(x,2));
end