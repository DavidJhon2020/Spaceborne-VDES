clear;clc;tic;

times=2000;    %??·′?y
N=3;    %?μà?
% K=1;
M=2;    %混合信号数目
nsamp =50;     %每比特采样点敿
colli=0;    %碰撞起始位置
an=0.5; %接收信号幅忍
% SNR=0; %信噪毿
% BER=zeros(times*21,N);
BER_DATA=zeros(times*21,N);%0:0.5＿0
RHO_DATA=zeros(times*21,N);
berquxian=zeros(1,21);
rhoquxian=zeros(1,21);
for SNR=0:0.5:10

for iii=1:times


fitness=0;
% BER=zeros(times,N);
% BER_DATA=zeros(times,N);
% RHO=zeros(times,N);
%A=[0.8 0.6 0.4;0.6 0.4 0.2];    %混合矩阵  


L = 4;      %Gaussian pulse duration for one bit 
Tb = 1;      %One bit time supposed to be 1
Ts = Tb/nsamp;   %Sampling period采样间隔
Fc1=161.975*10^1;   %仿真载波频率
Fb=1/Tb;
Bt=0.4;
Bb=Bt*Fb;
f0=2*pi*Fc1;
data_posedge = [1 1 1 1 1 1 1 1];   % 上升沿
data_training_seq = [ 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 ];  %训练序列
data_start_flag = [0 1 1 1 1 1 1 0];   % 弿§?标志 
data_stop_flag = [0 1 1 1 1 1 1 0];   % 结束标志 
data_fcs = zeros(1,16); %帧校验序刿  
data_buffer = zeros(1,24); % 缓冲使
data_interval=zeros(1,44);
h=0.5;

% for j=1:N
%     data_payload(j,:) = randi([0,1],1,168);%随机生成数据
% %     data_payload(j,:) =[1 1 1 1 0 0 1 0 1 0 0 0 1 1 1 1 1 0 0 1 1 ...
% %                         0 0 1 1 1 0 1 0 0 1 1 0 0 1 1 1 0 0 0 1 1 ...
% %                         0 0 0 0 0 1 1 1 1 1 0 0 1 1 0 0 0 1 1 0 0 ...
% %                         1 1 1 0 0 0 1 1 0 0 1 1 1 1 0 0 1 1 0 0 0 ...
% %                         1 1 0 0 0 0 1 1 0 0 1 1 1 0 0 0 0 0 1 1 0 ...
% %                         0 0 1 1 1 0 0 0 1 1 0 0 1 1 0 0 1 1 0 0 1 ...
% %                         1 0 0 0 0 1 1 1 1 0 0 0 1 1 1 0 0 1 1 0 0 ...
% %                         0 0 1 1 1 0 0 1 1 1 1 1 1 1 0 0 0 0 0 1 1];
%         
% %数据组帧
%     data_m(j,:) = [data_posedge data_training_seq data_start_flag  data_payload(j,:) data_fcs data_stop_flag data_buffer ];%长度256bit data_paylosd排在笿1~208
% end

data_m=[1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,0,0,0,1,0,0,1,0,1,1,1,0,0,1,0,1,1,0,1,0,0,1,0,1,1,1,0,1,1,0,0,1,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,1,0,1,0,0,1,0,0,1,0,0,0,1,1,1,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0,0,1,0,1,0,0,1,1,1,1,1,1,1,0,0,1,0,1,0,1,1,1,1,1,0,1,0,1,0,1,1,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,0,0,1,1,1,0,0,0,1,1,1,1,1,0,0,1,0,0,0,1,0,1,1,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,1,0,1,1,0,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,0,1,1,1,0,0,1,0,0,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,1,0,1,1,0,0,1,1,0,0,1,0,1,0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,0,1,1,0,0,0,1,0,1,0,1,1,1,1,0,1,0,1,1,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;1,1,1,1,1,1,1,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,1,1,1,1,1,0,1,0,1,0,0,1,1,1,1,1,1,0,1,0,1,0,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0,1,0,0,1,0,0,0,0,1,0,1,1,0,1,1,1,0,0,0,1,1,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,1,0,1,1,1,0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1,1,0,1,1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,0,1,1,0,1,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
for j=1:N

    x0 = zeros(size(data_m,1),size(data_m,2));
    x0(j,1) = 1;
    for i= 1:size(data_m,2)-1
        x0(j,i+1) = mod(x0(j,i) + data_m(j,i),2);
    end 
    %双极性编砿
end
    data_md =[1,-1,1,-1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,-1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,-1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,-1,1,-1,1,1,1,-1,-1,1,1,-1,1,-1,1,-1,-1,1,1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1;1,-1,1,-1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,-1,1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,1,-1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,-1,1,-1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,-1,1,1,1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,-1,1,1,1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,-1,1,-1,1,-1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1;1,-1,1,-1,1,-1,1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,-1,-1,1,1,1,-1,1,-1,1,-1,1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,1,1,1,1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,1,1,1,1,-1,1,1,-1,-1,-1,1,1,1,1,1,-1,-1,1,-1,-1,1,-1,1,1,1,1,-1,1,1,1,1,1,1,-1,1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,1,-1,-1,-1,1,1,1,1,-1,-1,-1,1,1,1,1,-1,-1,1,-1,-1,1,1,1,1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1];   
for j=1:N

    
    %内插
    data_md1(j,:)=upsample(data_md(j,:),nsamp);

%-------------------------------GMSK调制--------------------------------------------------------------------
    %高斯滤波器系敿
    dt=-L/2*Tb:Tb/nsamp:L/2*Tb;   
    gausst=(erfc(2*pi*Bb*(dt-Tb/2)/sqrt(2)/sqrt(log(2)))-erfc(2*pi*Bb*(dt+Tb/2)/sqrt(2)/sqrt(log(2))))/4/nsamp;   
    %公式2.8
    
    %卷积
    dt_phase(j,:)=conv(data_md1(j,:),gausst,'same');
    L1=length(dt_phase(j,:));
    k=1:1:L1;
    t=(k-1)*Ts;
    phase(j,:)=2*pi*h*cumsum(dt_phase(j,k));%%GMSK调制的相位信恿公式2.9

    data_mod1(j,:)=cos(2*pi*Fc1*(k-1)*Ts+phase(j,:));%经过GMSK调制的发射信叿
    
    N_phase=length(phase(j,:));
    N_all=N_phase;
    
    data_xs(j,:)=zeros(1,(256*N*nsamp-(256-colli)*(N-1)*nsamp));%接收信号初始化？？？？？？？？？？？？？？？？？？？？？？？？？？？＿
    
    
%         data_xs(j,(190*(j-1)+1):(190*(j-1)+256))=data_mod1(j,:);  
    data_xs(j,(colli*(j-1)*nsamp+1):(colli*(j-1)+256)*nsamp)=an^(j-1)*data_mod1(j,:);  %接收信号
    data_xs1(j,:)=awgn(data_xs(j,:),SNR,'measured');    %加噪的接收信叿
 %---------------------------------------------------------------------------------------------------------------
 

end
% figure(1);
%     plot(data_md(1,:) );
%     axis([30,150,-1.1,1.1]);
%     xlabel('n');
%     ylabel('b(n)');
%     title('差分编码后序刿);
%     
%     figure(7);
%     plot(dt_phase(1,:) );
%     axis([30*3,150*3,-0.2,0.2]);
% 
%     xlabel('t');
%     ylabel('dt_phase');
%     title('高斯滤波器输出波彿);

%    figure(8);
%    plot(data_mod1(1,:) );
%    axis([30*50,33*50,-1.1,1.1]);
%    xlabel('t');
%    ylabel('data_mod1');
%    title('中频调制信号');
%    
%     
%    figure(11);
%    plot(phase(1,:) );
% 
%    xlabel('t');
%    ylabel('dt_phase');
%    title('GMSK的调制相位波彿);

%接收信号
s=data_xs1;

%绘制接收信号囿
% figure(2);
% subplot(3,3,1),plot(s(1,:)),title('输入信号1');
% subplot(3,3,4),plot(s(2,:)),title('输入信号2');
% subplot(3,3,7),plot(s(3,:)),title('输入信号3');

%生成混合矩阵
% srand=rand(size(s,1));%chen:3*3
 srand=[0.989887419685116,0.738943220511530,0.763624579930168;0.710968887412684,0.647694041732414,0.695839741809154;0.767587562804382,0.502637780501000,0.159953030223195];

mixeds_1=srand*s;%冲突信号
for i=1:3
    mixeds(i,:)=mixeds_1(i,:);
end

%绘制冲突信号囿
% figure(3);
% subplot(3,3,2),plot(mixeds(1,:)),title('混合信号1');
% subplot(3,3,5),plot(mixeds(2,:)),title('混合信号2');
% subplot(3,3,8),plot(mixeds(3,:)),title('混合信号3');
mixeds_bak=mixeds;

%--------------------改进的FastICA-----------------------------------
%标准匿
mixeds_mean=zeros(3,1);
mixeds_mean=mean(mixeds,2);%chen:求各行的均忍
x=ones(1,size(s,2));
y=mixeds_mean*x;%wang:x,y用来转化为相同行列数的矩阵才能进行下步的减法计算
mixeds_mean=mixeds-y;
%白化
mixeds_cov=cov(mixeds_mean');%chen:计算信号mixeds_mean方差
[E,D]=eig(mixeds_cov);%chen：[V,D] = eig(A)返回矩阵的特征忥??特征向釿D是特征忯??V是特征向釿
q=inv(sqrt(D))*(E)';%chen：?星载AIS接收机的多用户分离算法研究??-公弿-28  白化矩阵
mixeds_white=q*mixeds_mean;
%wang:IsI=cov(mixeds_white');

%改进的fastICA算法
 x=mixeds_white;
[v,m]=size(x);
numoica=v;
B=zeros(numoica,v);%分离矩阵初始匿
for r=1:numoica
    j=1;
    P=400;
%     b=rand(numoica,1)-.5;%chen：分离向量随机赋初忍
%     b=b/norm(b);%chen：分离向量归丿??，norm返回向量b的二范数，即樿
%     t=x'*b;
%     G=-exp(-t.^2/2);

%------------------应用粒子群算法初步寻伿--------------------
%     [b,fitness]=PSO_algo_forICA(x,30,1.5,1.5,0.8,50,N);    

%------------------应用布谷鸟算法初步寻伿--------------------
% b=1+(-2)*rand(1,3);
b=[-0.103373265099220,-0.254805127199537,0.669756511825919];
ZJ=b;
b=b';
mt=zeros(N,1);%是初始化为零值还是零矩阵＿初始化为零向釿
vt=zeros(N,1);  
elta=0.001; 
beta1=0.5;%β1 0.9 丿??矩估计的指数衰减玿
beta2=0.75;
    while j<=P
        t=x'*b;
        g=t.^3;
        dg=3*t.^2;
        b=x*g/m-mean(dg)*b;%  fastICA核心公式＿chen:《FASTICA-1999TNN-Fast and Robust Fixed-Point Algorithms for Independent Component Analysis》公弿5    
%         wt=b;
%     mt=beta1*mt+(1-beta1)*wt; %噪声梯度or指数移动均忬beta1 咿beta2 控制了这些移动均值的衰减玿
%     vt=beta2*vt+(1-beta2)*(wt.*wt);%平方梯度or有偏方差
%     mt_bias=mt./(1-power(beta1,j));%1-β1^t 偏差修正（修正很微小＿
%     vt_bias=vt./(1-power(beta2,j));%代表丿?a近似对角线的费舍尔信息矩阿
%     wt=wt-elta*(mt_bias./(sqrt(vt_bias)));%
%         b=wt/norm(wt);    
        b=b/norm(b);   
        B(:,r)=b;
        j=j+1;
    end
end
%ICA的数据还县
icaeds=B'*q*mixeds_bak;%分离矩阵*白化矩阵*未去均忦··合信号

% %绘制分离信号囿
% figure(4);
% subplot(4,3,7),plot(icaeds(1,:)),title('ica解混信号1');
% subplot(4,3,8),plot(icaeds(2,:)),title('ica解混信号2');
% subplot(4,3,9),plot(icaeds(3,:)),title('ica解混信号3');

%wang:Rho1进行顺序判断重排
x_rec1=zeros(N,length(icaeds(1,:)));
Rho1=zeros(N,N);
for ii=1:N
    for jj=1:N
        Rho1(ii,jj)=correcf(s(ii,:),icaeds(jj,:));
        if abs(Rho1(ii,jj))>0.5
           if icaeds(jj,340)/s(ii,340)>0
                x_rec1(ii,:)=icaeds(jj,:);
            else
                x_rec1(ii,:)=-icaeds(jj,:);
            end
        end
    end
end

%按序绘制分离信号囿
% figure(5);
% subplot(3,3,3),plot(x_rec1(1,:)),title('分离承??信号1');
% subplot(3,3,6),plot(x_rec1(2,:)),title('分离承??信号2');
% subplot(3,3,9),plot(x_rec1(3,:)),title('分离承??信号3');

%--------------Differential coherent demodulation--------------------------
SL_test = LPF(2*f0,x_rec1(2,((2-1)*colli*nsamp+1):(colli*(2-1)+256)*nsamp),21); 
SL=zeros(N,length(SL_test));
diff_out = zeros(N,N_all-2*nsamp+1);
Len=floor((size(diff_out,2)-1)/nsamp)*nsamp+1;
diffout_dw=zeros(N,(Len-1)/nsamp+1);
L1=length(diffout_dw(1,:));
diff_data=zeros(N,L1);

numerr=zeros(1,N);
numerr_data=zeros(1,N);

ber=zeros(1,N);
ber_data=zeros(1,N);


for i2=1:N
    SL(i2,:) = LPF(2*f0,x_rec1(i2,((i2-1)*colli*nsamp+1):(colli*(i2-1)+256)*nsamp),21);   %对分离信号低通滤泿
    for i = 1:N_all-2*nsamp+1
        diff_out(i2,i) = sum(SL(i2,i:i+nsamp-1).*SL(i2,i+nsamp:i+2*nsamp-1));
    end
    diffout_dw(i2,:)=diff_out(i2,1:nsamp:Len);
    for i=1:L1    %判决
        if diffout_dw(i2,i)>0
            diff_data(i2,i)=1;
        elseif diffout_dw(i2,i)<0
            diff_data(i2,i)=-1;
        end
    end
    
%     
%     for i=2:256
%         j=i-1;
% %     flag=abs(diff_data(j)-data_md1(1,i));
%         flag=abs(diff_data(i2,j)-data_md(i2,i));
%         if (flag~=0)
%             numerr(i2)=numerr(i2)+1;
%         end
%     
%     end
    for i=42:209
        j=i-1;
%     flag=abs(diff_data(j)-data_md1(1,i));
        flag=abs(diff_data(i2,j)-data_md(i2,i));
        if (flag~=0)
            numerr_data(i2)=numerr_data(i2)+1;
        end
    
    end
%     ber(i2)=numerr(i2)/256;   %误码率计箿
    ber_data(i2)=numerr_data(i2)/168;
end   


for j=1:N
    Rho(j)=correcf(s(j,(colli*(j-1)*nsamp+1):(colli*(j-1)+256)*nsamp),x_rec1(j,(colli*(j-1)*nsamp+1):(colli*(j-1)+256)*nsamp));
end 

BER_DATA(2*times*SNR+iii,:)=ber_data;
RHO_DATA(2*times*SNR+iii,:)=Rho;

end
for xx=(2*times*SNR+1):(2*times*SNR+times)
    berquxian(1,SNR*2+1)=berquxian(1,SNR*2+1)+BER_DATA(xx,1);
    rhoquxian(1,SNR*2+1)=rhoquxian(1,SNR*2+1)+RHO_DATA(xx,1);   
end
berquxian(1,SNR*2+1)=berquxian(1,SNR*2+1)./times;
rhoquxian(1,SNR*2+1)=rhoquxian(1,SNR*2+1)./times;
end
figure;
i3=0:0.5:10;
semilogy(i3,berquxian);
grid on;
hold on;

figure;
MM=linspace(0,10);
NN=spline(i3,berquxian,MM);
semilogy(MM,NN);
grid on;
hold on;

figure;
PP=linspace(0,10);
QQ=spline(i3,rhoquxian,PP);
plot(PP,QQ,'r',i3,rhoquxian,'*');
grid on;
hold on;

