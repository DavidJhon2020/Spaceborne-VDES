function ObjVal=kurt(w,z,K,N)
for i=1:K
    A=w(i,:);
%     w1=zeros(N,N);
%     w2=zeros(N,N);
%     w3=zeros(N,N);
    [W1,W2,W3,W4,W5,W6]=givens(A);
    W=W1*W2*W3*W4*W5*W6;
      for p=1:N         % 向量单位化
       W(:,p)=W(:,p)/norm(W(:,p));
      end
    y=(W)'*z;
%     g=y.^2;
%     G=y.^4;
%     G1=mean(G(:));
%     G2=mean(g(:));
%     ObjVal(1,i)=abs(G1-3*((G2).^2));  
    %J1=mean(tanh(y(:)));
    J1=mean(log(cosh(y(:))));
    y_guass=normrnd(0,1,4,12800);
   %J2=mean(tanh(y_guass(:))); 
    J2=mean(log(cosh(y_guass(:))));
    J=(J1-J2)^2;
    ObjVal(1,i)=J;
end


% hang=size(w,1);
% if hang==K
% for i=1:K
%     A=w(i,:);
%     W=reshape(A,N,N);
%      for p=1:N         % 向量单位化
%       W(:,p)=W(:,p)/norm(W(:,p));
%      end
%     W=(W*W')^(-0.5)*W;% 矩阵正交化
%     y=(W)'*z;
%     g=y.^2;
%     G=y.^4;
%     G1=mean(G(:));
%     G2=mean(g(:));
%     ObjVal(1,i)=abs(G1-3*((G2).^2));
% end
%  else
%  W=reshape(w,N,N);
%      for p=1:N         % 向量单位化
%       W(:,p)=W(:,p)/norm(W(:,p));
%      end
%     W=(W*W')^(-0.5)*W;% 矩阵正交化
%     y=(W)'*z;
%     g=y.^2;
%     G=y.^4;
%     G1=mean(G(:));
%     G2=mean(g(:));
%     ObjVal=abs(G1-3*((G2).^2)); 
% end
