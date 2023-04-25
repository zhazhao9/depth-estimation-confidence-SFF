function FM = fmeasure(Image,Measure,WSize)
%This function measures the relative degree of focus of
%an image. It may be invoked as:
%
%   FM = focusmeasure(Image, Measure, WSize)
%
%Where
%   Image,  is a DOUBLE Image and FM is a
%           matrix the same size as Image with the
%           computed focus measure for every pixel.
%   Measure, is the focus measure algorithm as a string.
%   WSize,  is the size of the neighborhood used to
%           compute the focus value of every pixel.
%
MEANF = fspecial('average',[WSize WSize]); %¾ùÖµÂË²¨

switch Measure
    case 'Brenner' % Brenner's (Santos97)
        [M, N] = size(Image);
        DH = zeros(M, N);
        DV = zeros(M, N);
        DV(1:M-2,:) = Image(3:end,:)-Image(1:end-2,:);
        DH(:,1:N-2) = Image(:,3:end)-Image(:,1:end-2);
        FM = max(DH, DV);
        FM = FM.^2;
        FM = imfilter(FM, MEANF,'replicate');
        
    case 'SML' %Sum of Modified Laplacian (Nayar89)
        M = [-1 2 -1];
        Lx = imfilter(Image, M, 'replicate', 'conv');
        Ly = imfilter(Image, M', 'replicate', 'conv');
        FM = abs(Lx) + abs(Ly);
        FM = imfilter(FM, MEANF, 'replicate');
        
    case 'Ten' % Tenengrad (Krotkov86)
        Sx = fspecial('sobel');
        Gx = imfilter(Image, Sx, 'replicate', 'conv');
        Gy = imfilter(Image, Sx', 'replicate', 'conv');
        FM = Gx.^2 + Gy.^2;
        FM = imfilter(FM, MEANF, 'replicate');
        
        
    case 'GLVA' % Graylevel variance (Krotkov86)
        FM = stdfilt(Image, ones(WSize,WSize)).^2;
        FM = imfilter(FM, MEANF, 'replicate');

        
    case 'Ten4DIR' 
        Sx = fspecial('sobel');
        Tx = [-2 -1 0;-1 0 1;0 1 2];
        Gx = imfilter(Image, Sx, 'replicate', 'conv');
        Gy = imfilter(Image, Sx', 'replicate', 'conv');
        Hx = imfilter(Image, Tx, 'replicate', 'conv');
        Hy = imfilter(Image, flipud(Tx), 'replicate', 'conv');
        FM = Gx.^2 + Gy.^2 + Hx.^2 + Hy.^2;
        FM = imfilter(FM, MEANF, 'replicate');
        
    case 'DCTR' % DCT reduced energy ratio (Lee2009)
        FM = nlfilter(Image, [8 8], @ReRatio);
        FM = imfilter(FM, MEANF, 'replicate');
        
    case 'WAVS' %Sum of Wavelet coeffs (Yang2003)
        [C,S] = wavedec2(Image, 1, 'db6');
        H = wrcoef2('h', C, S, 'db6', 1);
        V = wrcoef2('v', C, S, 'db6', 1);
        D = wrcoef2('d', C, S, 'db6', 1);
        FM = abs(H) + abs(V) + abs(D);
        FM = imfilter(FM, MEANF, 'replicate');
        
        
end


%*********************@ReRatio***************************
function fm = ReRatio(M)
M = dct2(M);
fm = (M(1,2)^2+M(1,3)^2+M(2,1)^2+M(2,2)^2+M(3,1)^2)/(M(1,1)^2);
