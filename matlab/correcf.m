function [ Rho ] = correcf( S1,S2 )
%correlation coefficient
%  
T=size(S1,2);
S1mean=0;
S1sum=0;
S2mean=0;
S2sum=0;
for t=1:T
    S1sum=S1sum+S1(t);
    S2sum=S2sum+S2(t);
end     
S1mean=S1sum/T;
S2mean=S2sum/T;
    
sum1=0;
sum2=0;
sum3=0;
for t=1:T
    sum1=sum1+(S1(t)-S1mean)*(S2(t)-S2mean);
    sum2=sum2+(S1(t)-S1mean)^2;
    sum3=sum3+(S2(t)-S2mean)^2;
end
    
sqr2=sqrt(sum2);
sqr3=sqrt(sum3);
    
Rho=sum1/(sqr2*sqr3);


end

