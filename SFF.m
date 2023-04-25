%% 清屏、清除所有变量和窗口
clc
clear;
close all;

%% 加载图像数据

fprintf('Reading      ')
load CosGP005.mat
load CosD.mat
img = CosGP005;
gt = CosD;
rows = size(img,1);
cols = size(img,2);
sz = [rows,cols];
numframes = size(img,3);
focus = 1:numframes;
% focus = linspace(0.5,0.05,100);

% show images
for i = 1:numframes
    imshow(img(:,:,i),[]);
    fprintf('\b\b\b\b\b[%2.2i%%]',round(100*i/numframes))
    pause(0.02)
end
close all



% fprintf('Reading      ')        %read images from files
% filepath = 'C:\Users\ZMQZZ\Documents\学习\dataset\1yuanO';
% addpath(filepath);
% filelist = dir(filepath);
% filename = filelist(3).name;
% [~,name,ext] = fileparts(filename);
% name = name(1:end-1);
% image = imread(filename);
% rows = size(image,1);
% cols = size(image,2);
% sz = [rows,cols];
% numframes = length(dir(filepath))-2; 
% img = zeros(rows,cols,numframes);
% 
% focus = 1:numframes;
% % focus = linspace(1500,0,151);
% for i = 1:numframes
%     I=imread([name,int2str(i),ext]);
%     I=im2double(I);
%     img(:,:,i) = I;
%     imshow(I,[]);
%     fprintf('\b\b\b\b\b[%2.2i%%]',round(100*i/numframes))
%     pause(0.02)
% end
% close all

% %焦栈间隔
% delta = 50.50;

%% Focus Volume
fprintf('\nFmeasure      ')
fm = zeros(rows,cols,numframes);
for i = 1:numframes     
    Image = img(:,:,i);
    fm(:,:,i) = fmeasure(Image,'Ten',5);  
    fprintf('\b\b\b\b\b[%2.2i%%]',round(100*i/numframes))
end
fprintf('\n')


%% Pearson's correlation coefficient  
fprintf('\nPearson correlation coefficient      ')
mean_fm = mean(fm,3);
mean_im = mean(img,3);
fm_D = fm-mean_fm;
im_D = img-mean_im;
up = sum(fm_D.*im_D,3)/(numframes-1);

std_fm = std(fm,0,3);
std_im = std(img,0,3);
sub = std_fm.*std_im;

pcc = abs(up./sub);
pcc(isnan(pcc))=0.01;


%% cosine similarity
fprintf('\nCosine similarity      ')
[I,J,K]=size(img);
cs=zeros(I,J);

    for i=1:I
        parfor j=1:J
            gray_curve = mapminmax(squeeze(img(i,j,:))',0,1);
            gray_curve_rect=abs(gray_curve-gray_curve(1));
            fm_curve = mapminmax(squeeze(fm(i,j,:))',0,1);
            cs(i,j)=1-pdist2(gray_curve_rect,fm_curve,'cosine');
        end
        fprintf('\b\b\b\b\b[%2.2i%%]',round(100*i/I))
    end
delete(gcp);
cs(isnan(cs))=0.01;

%% rms
rms_mat = zeros(sz);
delete(gcp);
parpool(4)
for i=1:rows
    parfor j=1:cols
        gray_curve = mapminmax(squeeze(img(i,j,:))',0,1);
        gray_curve = abs(gray_curve-gray_curve(1));
        fm_curve = mapminmax(squeeze(fm(i,j,:))',0,1);
        rms = rms(gray_curve)-rms(fm_curve);
        rms_mat(i,j) = rms;
    end
    fprintf('\b\b\b\b\b[%2.2i%%]',round(100*i/rows))
end

%% similarity + rms
newpcc = pcc+rms_mat;
newcs = cs+rms_mat;
%% Estimate depthmap 

fprintf('\nDepthmap ')

%三点拟合
[I, zi, ~, ~] = gauss3P(focus, fm);
zi(zi>max(focus)) = max(focus);
zi(zi<min(focus)) = min(focus);

fmax = focus(I);
zi(isnan(zi)) = fmax(isnan(zi));

fprintf('[100%%]\n' )

%% GIF using G map

I1 = newpcc;
I2 = newcs;
p = zi;
r = 61;
eps = 0.8^2; % try eps=0.1^2, 0.2^2, 0.4^2
q1 = guidedfilter(I1, p, r, eps);
q2 = guidedfilter(I2, p, r, eps);


%% Display the result:

surf(zi), shading Interp , colormap((jet))
set(gca, 'zdir','normal','xdir','normal','ydir','normal')

% figure,surf(zc), shading Interp , colormap jet
% set(gca, 'zdir','normal',   'xdir','normal','ydir','reverse')

% 
% figure,surf(zp), shading Interp , colormap jet
% set(gca, 'zdir','normal','xdir','normal','ydir','reverse')
% 
% figure,surf(gt), shading Interp , colormap((jet))
% set(gca, 'zdir','normal','xdir','normal','ydir','normal')

figure,surf(q1), shading Interp , colormap((jet))
set(gca, 'zdir','normal','xdir','normal','ydir','normal')
figure,surf(q2), shading Interp , colormap((jet))
set(gca, 'zdir','normal','xdir','normal','ydir','normal')

%% MSE、SNR、PSNR、SSIM

ref = gt;
tes1 = q1;
tes2 = q2;
    
fprintf('\n ');
RMSE1 = sqrt(immse(tes1,ref));
fprintf('\n The RMSE value of pcc is %0.4f', RMSE1);
CORR1 = corr2(tes1,ref);
fprintf('\n The CORR value of pcc is %0.4f', CORR1);
fprintf('\n ');
RMSE2 = sqrt(immse(tes2,ref));
fprintf('\n The RMSE value of cs is %0.4f', RMSE2);
CORR2 = corr2(tes2,ref);
fprintf('\n The CORR value of cs is %0.4f', CORR2);

%% Function
function [I, u, s, A] = gauss3P(x, Y)
% Closed-form solution for Gaussian interpolation using 3 points
% Internal parameter:()
STEP = 2;
%%%%%%%%%%%%%%%%%%%
[M,N,P] = size(Y);      
[~, I] = max(Y,[ ], 3);    
[IN,IM] = meshgrid(1:N,1:M);      
Ic = I(:);  
Ic(Ic<=STEP)=STEP+1;      
Ic(Ic>=P-STEP)=P-STEP;    
Index1 = sub2ind([M,N,P], IM(:), IN(:), Ic-STEP);      
Index2 = sub2ind([M,N,P], IM(:), IN(:), Ic);
Index3 = sub2ind([M,N,P], IM(:), IN(:), Ic+STEP);
% Index1(I(:)<=STEP) = Index3(I(:)<=STEP);
% Index3(I(:)>=STEP) = Index1(I(:)>=STEP);
x1 = reshape(x(Ic(:)-STEP),M,N);
x2 = reshape(x(Ic(:)),M,N);
x3 = reshape(x(Ic(:)+STEP),M,N);
y1 = reshape(log(Y(Index1)),M,N);
y2 = reshape(log(Y(Index2)),M,N);
y3 = reshape(log(Y(Index3)),M,N);

c = ( (y1-y2).*(x2-x3)-(y2-y3).*(x1-x2) )./...
    ( (x1.^2-x2.^2).*(x2-x3)-(x2.^2-x3.^2).*(x1-x2) );
b = ( (y2-y3)-c.*(x2-x3).*(x2+x3) )./(x2-x3);
a = y1 - b.*x1 - c.*x1.^2;

s = sqrt(-1./(2*c));
u = b.*s.^2;
A = exp(a + u.^2./(2*s.^2));
end

