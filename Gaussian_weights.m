function S=Gaussian_weights(X,m,n)

g=10;

[x,y]=size(X);
area=x+y;
sigma=ceil(area/g);
S=zeros(x,y);
for i=1:x
    for j=1:y
        S(i,j)=(1/(2*pi*sigma^2))*exp(-((i-m)^2+(j-n)^2)/(2*sigma^2));

    end
end

end




