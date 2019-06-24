function o = rootExpandKernelMap(x)
    s = sign(x);
    y = (s.*x).^0.5;
    o = [y.*(s==1) y.*(s==-1)]; 
     
end