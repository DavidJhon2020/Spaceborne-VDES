function ObjVal=kurts(w,z,K,N)
    [W1,W2,W3,W4,W5,W6]=givens(w);
    W=W1*W2*W3*W4*W5*W6;
      for p=1:N         % 向量单位化
       W(:,p)=W(:,p)/norm(W(:,p));
      end
    y=(W)'*z;
%     g=y.^2;
%     G=y.^4;
%     G1=mean(G(:));
%     G2=mean(g(:));
%     ObjVal=abs(G1-3*((G2).^2));  
%     J1=mean(tanh(y(:)));
    J1=mean(log(cosh(y(:))));
    y_guass=normrnd(0,1,4,12800);
%     J2=mean(tanh(y_guass(:)));
    J2=mean(log(cosh(y_guass(:))));
    J=(J1-J2)^2;
    ObjVal=J;
end