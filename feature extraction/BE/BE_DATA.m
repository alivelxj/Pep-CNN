clear all
clc
input=importdata('F:\python\KNN\SBP_test.txt');
data=input(2:2:end,:);
[m,n]=size(data);
label1=ones(1258,1);
label2=zeros(1887,1);
label=[label1;label2];
out=[];
input=data;
for i=1:m
    protein=input{i};
    output =BE_feature(protein);
    out=[out;output];
    ouput=[];
end
save BE_SBP_test.mat out