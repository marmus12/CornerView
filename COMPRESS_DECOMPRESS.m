clear all
close all
%% CONFIGURATION
%/media/emre/Data/cepinet_runs/08-05-12-40_greek/tests/
%0034__10-05-15-19__all_corners__hci/pfms/corner_warping_14-05-15-13/
%combine_warps_31-05-18-05
samples = {'vinyl','kitchen','museum','greek','dino','dots','bedroom'...
            ,'pyramids','stripes','bicycle','backgammon'...
            ,'origami','boxes','cotton','sideboard','herbs'};
% samples = { 'vinyl',  'kitchen', 'museum','greek'};

save_dir = '/home/emre/Desktop/deneme/';

% ours:
center_pfm_dir = '/media/emre/Data/epinet_runs/09-05-13-05_greek/tests/0054__31-05-12-36__hci/pfms/'; 
corner_pfm_dir = '/media/emre/Data/cepinet_runs/08-05-12-40_greek/tests/0034__10-05-15-19__all_corners__hci/pfms/'; 
comb_warps_dir = [corner_pfm_dir 'corner_warping_14-05-15-13/combine_warps_19-06-18-55/'] 
num_target_pixels = (490*490);
%%%


% % % gt disps:
% center_pfm_dir = '/media/emre/Data/heidelberg_full_data/additional/additional_depth_disp_all_views/gt_dips/'; 
% corner_pfm_dir = '/media/emre/Data/heidelberg_full_data/additional/additional_depth_disp_all_views/gt_dips/'; 
% comb_warps_dir = [corner_pfm_dir 'corner_warping_20-06-16-34/combine_warps_24-06-10-26/'] 
% num_target_pixels = (512*512);
% %%%

ds = 'hci';
comprss_bref_with = 'cerv'; %'jp2k';%'cerv'; %'jp2k'; %'cerv';%'jp2k'

if strcmp(comprss_bref_with,'cerv')
    cerv_dir = '/home/emre/Documents/kodlar/JOURNAL_WORK/';
    cerv_temp_dir = '/home/emre/Desktop/deneme4/';
    cerv_bs_name = 'segm.bs'; 
    addpath(cerv_dir);
end
%%



if strcmp(ds,'hci')
    ds_dir = '/media/emre/Data/heidelberg_full_data/';
    corner_codes = {'0_0','0_8','8_0','8_8'};
    center_code = '4_4';
    ref_im_codes = {'000','008','040','072','080'}; 
    
    num_LF_pixels = (9*9*num_target_pixels);
end    
% elseif strcmp(ds,'lytro')
%    ds_dir = '/media/emre/Data/Lytro_lenslet/';
%    corner_codes = {'3_3','3_11','11_3','11_11'};
%    center_code = '7_7'; 
%    ref_im_codes = {'003_003','003_011','007_007','011_003','011_011'}; 
%    num_target_pixels = (604*412);
%    num_LF_pixels = (15*15*num_target_pixels);   
% end

%% CODELENGTH of REF COLOR IMS (5 for all targets)

num_refs = numel(ref_im_codes);

num_corners = numel(corner_codes);

ref_codes = [corner_codes center_code];

for sample_ind = 1:length(samples)
    sample = samples{sample_ind};
    sample_dir = dir([ds_dir '*/' sample]);
    sample_folder = [sample_dir(2).folder '/'];

    total_ref_im_bits = 0;
    for ref_im_ind = 1:num_refs

        ref_im_code = ref_im_codes{ref_im_ind};
        ref_im_path = [sample_folder 'input_Cam' ref_im_code '.png'];
        ref_im = imread(ref_im_path);

        jp2k_file = [save_dir 'ref_im.jp2'];    
        imwrite(ref_im,jp2k_file,'Mode','lossless');
        file_info = dir(jp2k_file);
        jp2k_bits = file_info.bytes*8;

        total_ref_im_bits = total_ref_im_bits + jp2k_bits;
    end
    %% CODELENGTH of Estimated Ref Disparities (5 for all targets)

    total_D_bits = 0;
    for corner_ind = 1:num_corners
        corner_code = corner_codes{corner_ind};
        pfm_path = [corner_pfm_dir sample '_' corner_code '.pfm'];
        D = pfmread(pfm_path);
        jp2k_file = [save_dir 'D.jp2'];    
        imwrite(D,jp2k_file,'Mode','lossless');
        file_info = dir(jp2k_file);
        jp2k_bits = file_info.bytes*8;    
        total_D_bits = total_D_bits + jp2k_bits;
    end

    pfm_path = [center_pfm_dir sample '_' center_code '.pfm'];
    D = pfmread(pfm_path);
    jp2k_file = [save_dir 'D.jp2'];    
    imwrite(D,jp2k_file,'Mode','lossless');
    file_info = dir(jp2k_file);
    jp2k_bits = file_info.bytes*8;    
    total_D_bits = total_D_bits + jp2k_bits;

    %% Best Labels Images (1 for each target)
    target_codes = {};
    for iar = 0:8
        for iac = 0:8
            target_code = [num2str(iar) '_' num2str(iac)];
            if ~any(strcmp(ref_codes,target_code))
                target_codes = [target_codes target_code]; 
            end      
        end
    end
    all_codes = [target_codes ref_codes];
    num_targets = numel(target_codes);

    total_best_ref_im_bits = 0;
    for target_ind = 1:num_targets
        target_code = target_codes{target_ind};
        best_ref_im_path = [comb_warps_dir sample '_' target_code '_best_ref_im.mat'];
        dummy = load(best_ref_im_path);
        best_ref_im = int8(dummy.bri);
        
        if strcmp(comprss_bref_with,'cerv')
%             compressed_file = ['/home/emre/Desktop/enc_best_ref.bs'];
            
            [NR,NC] = size(best_ref_im);
            scripts_dir = pwd;
            cd(cerv_dir)
            [CLcervBPP, CLcerv, ~, ~] = H_CERV_CONTOURS1(best_ref_im,NR,NC,cerv_temp_dir,cerv_bs_name);
            [~, ~, decoded_smap, ~] = H_CERV_CONTOURS1([],NR,NC,cerv_temp_dir,cerv_bs_name);
            cd(scripts_dir);        
            assert(isequal(decoded_smap,best_ref_im));
            file_info = dir([cerv_temp_dir cerv_bs_name]);
            
        elseif strcmp(comprss_bref_with,'jp2k')
            compressed_file = [save_dir 'best_ref_im.jp2'];    
            imwrite(best_ref_im,compressed_file,'Mode','lossless');
            file_info = dir(compressed_file);
        end
        
        this_target_bits = file_info.bytes*8;
        total_best_ref_im_bits = total_best_ref_im_bits + this_target_bits;    
        target_best_ref_bits(target_ind) = this_target_bits;
    end
    %% Residual Images (1 for each target)

    total_res_im_bits = 0;
    for target_ind = 1:num_targets
        target_code = target_codes{target_ind};
        %convert target code: %% 
        flat_target_code = flatten_target_code(target_code);
        %%%%%%%%%%%%%%%%%%%%%%%
        GT_path = [sample_folder 'input_Cam' flat_target_code '.png'];
        GT = imread(GT_path);
        cropped_GT = center_crop(GT,size(best_ref_im));

        dummy = dir([comb_warps_dir sample '_' target_code '_region*.png']);
        estimation_path = [dummy.folder '/' dummy.name];
        estimation = imread(estimation_path);


        Dummy = estimation - cropped_GT;    
        for icomp = 1:3
            DiffIm(:,:,icomp) = Dummy(:,:,icomp)-min(min(Dummy(:,:,icomp)))+1;
        end

        imwrite(DiffIm,[save_dir 'DiffIm_' sample '_' target_code '.png']);
        imwrite(uint16(DiffIm),'DiffIm1.jp2','Mode','lossless');
        file_info = dir('DiffIm1.jp2');
        this_target_bits = file_info.bytes*8;    
        total_res_im_bits = total_res_im_bits + this_target_bits;
        target_res_bits(target_ind) = this_target_bits;
    end


    %% JP2K Baseline

    total_LF_im_bits = 0;
    for LF_im_ind = 1:numel(all_codes)%0:(numel(all_codes)-1)

        view_code = all_codes{LF_im_ind}; 
        LF_im_code = flatten_target_code(view_code);
        LF_im_path = [sample_folder 'input_Cam' LF_im_code '.png'];
        LF_im = imread(LF_im_path);
        jp2k_file = [save_dir 'LF_im.jp2'];

        imwrite(LF_im,jp2k_file,'Mode','lossless');
        file_info = dir(jp2k_file);
        target_bits = file_info.bytes*8;
        target_BPP = target_bits/num_target_pixels;
        total_LF_im_bits = total_LF_im_bits + target_bits;

        [iar,iac] = code2coords(view_code);
        JP2K_target_BPPs(sample_ind,iar+1,iac+1) = target_BPP; 
    end

    %% total bits required to transmit to reconstruct the whole thing with our method:
    targetwisefree_total_bits = total_ref_im_bits + total_D_bits;
    targetwisefree_BPP = targetwisefree_total_bits/num_LF_pixels;
    total_bits = total_res_im_bits + total_best_ref_im_bits + targetwisefree_total_bits;
    overall_BPP = total_bits/num_LF_pixels;
    target_BPPs_by_sample1(sample_ind,1)=overall_BPP;
    %% total bits required to transmit to reconstruct the whole thing with JP2K:
    total_LF_im_bits
    overall_BPP_JP2K = total_LF_im_bits/num_LF_pixels;
    JP2K_target_BPPs_by_sample1(sample_ind,1)=overall_BPP_JP2K ;
    %% targetwise BPPs:

    for target_ind = 1:num_targets
       target_code = target_codes{target_ind};
       [iar,iac] = code2coords(target_code);
       target_best_ref_BPP = target_best_ref_bits(target_ind)/num_target_pixels;
       target_res_BPP = target_res_bits(target_ind)/num_target_pixels;
       targetspecific_BPP = target_best_ref_BPP + target_res_BPP;
       target_BPP = targetwisefree_BPP + targetspecific_BPP;
       target_BPPs(sample_ind,iar+1,iac+1) = target_BPP;  
    end
    
    for ref_ind = 1:num_refs
       ref_code = ref_codes{ref_ind};
       [iar,iac] = code2coords(ref_code);
       ref_BPP =  targetwisefree_BPP;%total_ref_im_bits/num_LF_pixels;
       target_BPPs(sample_ind,iar+1,iac+1) = ref_BPP;
    end
    
    
    
end

%% SUMMARIZE RESULTS:
target_BPPs_by_viewpoint = round(squeeze(mean(target_BPPs,1)),2)
target_BPPs_by_sample = round(squeeze(mean(target_BPPs,[2,3])),2)
target_BPPs_by_sample1 
JP2K_target_BPPs_by_viewpoint = round(squeeze(mean(JP2K_target_BPPs,1)),2);
JP2K_target_BPPs_by_sample = round(squeeze(mean(JP2K_target_BPPs,[2,3])),2)
JP2K_target_BPPs_by_sample1









