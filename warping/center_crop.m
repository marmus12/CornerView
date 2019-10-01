function [crop] = center_crop(im,crop_size)

    [imw,imh,~] = size(im);
    cw = crop_size(1);
    ch = crop_size(2);
    
    dw = (imw - cw)/2;
    dh = (imh - ch)/2;

    crop = im((dw+1):(dw+cw),(dh+1):(dh+ch),:);
end
