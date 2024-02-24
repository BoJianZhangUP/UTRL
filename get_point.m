function [m1,n1]=get_point(Z,poi)
        [x,y]=size(Z);
        area=x*y;
        K=round(sum(sum(Z~=0)));
        p=reshape(Z,[1,area]);
        [q, ~] = sort(p,'descend');
        m=0;
        n=0;
switch poi
    case {'mean'}
       
        for i=1:K
            [X,Y]=find(Z==max(max(q(i))));
            if size(X,1)>1
                m=m+sum(X)/size(X,1);
                n=n+sum(Y)/size(Y,1);

            else
                m=m+X;
                n=n+Y;
            end
        end

        if size(m,1)>1
            m1=sum(m)/size(m,1);
            n1=sum(n)/size(n,1);
            m1=floor(m1/K);
            n1=floor(n1/K);
        else
            m1=floor(m/K);
            n1=floor(n/K);
        end
    case {'max'}

        [X,Y]=find(Z==max(max(q(K))));
        if size(X,1)>1
            m1=m+sum(X)/size(X,1);
            n1=n+sum(Y)/size(Y,1);

        else
            m1=m+X;
            n1=n+Y;
        end
end
end
