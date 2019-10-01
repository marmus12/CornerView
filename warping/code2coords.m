function [iar,iac] = code2coords(code)

   dummy = strsplit(code,'_');
   iar = str2num(dummy{1});
   iac = str2num(dummy{2});
end

