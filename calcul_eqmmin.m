function [ ntrainmin ] =calcul_eqmmin( eqm,epsilon )
L=length(eqm);
for i=1:L-1
    if abs(eqm(i+1)-eqm(i))<epsilon 
        break
    end
 
end
ntrainmin=i;
end

