function flat_target_code = flatten_target_code(target_code)
    dummy = strsplit(target_code,'_');
    iar = str2num(dummy{1});
    iac = str2num(dummy{2});    
    flat_ind = sub2ind([9,9],iac+1,iar+1) - 1;
    flat_target_code = pad(num2str(flat_ind),3,'left','0');
end





