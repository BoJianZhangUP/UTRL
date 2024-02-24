function features = apply_UTRL_aggregation(X,sd)
[H,W,K] = size(X);

X(X<0)=0;
features=zeros(1,K);

% Apply thresholding to STc
STc=sd./sd;

index=sd<(2*(median(sd))-mean(sd)/(1/2*min(sd)));

STc(index)=0;

if H<64 && W <64
    X = imresize(X, 64/min(H,W)); % Preventing feature maps from being too small when halved in size
end

for i=1:3

    X_1=imresize(X,[size(X,1)/(2^(i-1)),size(X,2)/(2^(i-1))]);% Multi-scale features(X_E,X,X_R)

    S_xy=X_1.*(permute(STc,[1,3,2]));
   
    SG=soft_attention_module(sum(S_xy,3));% SA Mechanism

    XS = X_1.*SG;

    H_xy=XS.*(permute(STc,[1,3,2]));

    HG=hard_attention_module(sum(H_xy,3));%  HA Mechanism
    XH = X_1.*HG;

    XX=XS+XH;

    Z = channel_attention_module(XX); %  CA Mechanism
    Z=permute(Z,[1,3,2]);
    XX_Z = XX.*Z;
    sum_XX_Z = reshape(sum(XX_Z,[1,2]),[1,K]);
    features=features+sum_XX_Z;

end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function SG= soft_attention_module(sum_S)

[xs,ys]=get_point(sum_S,'mean');

SG=Gaussian_weights(sum_S,xs,ys);
SG=sum_S.^2.*(SG);
z = sum(sum(SG.^2))^(1/2);
SG = (SG/z).^(1/2);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function HG= hard_attention_module(sum_H)

[xh,yh]=get_point(sum_H,'max');

HG=Gaussian_weights(sum_H,xh,yh);
HG=sum_H.^2.*HG;
z = sum(sum(HG.^2))^(1/2);
HG = (HG/z).^(1/2);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function channel_wt = channel_attention_module(XX)

[~,~,K] = size(XX);

e=1*10^(-5);

t=sum(max(XX)).*sum(max(XX)>0);

t=reshape(t,[1,K]);

channel_wt = sqrt((max(t)./(t+e)));

end
