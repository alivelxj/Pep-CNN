clear all
clc
input=importdata('F:\python\KNN\SBP_test.txt');
data=input(2:2:end,:);
[m,n]=size(data);
vector=[];
for i=1:m;
 vector=[vector;EBGW_yu(data{i})];
end
save EBGW_SBP_test.mat vector