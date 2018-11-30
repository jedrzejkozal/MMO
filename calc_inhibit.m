function inhibit = calc_inhibit(traning_data, Nfindings, Ndiseases, prior)

    inhibit = zeros(Ndiseases, Nfindings);
    sum_all = Ndiseases*Nfindings*length(traning_data);
    %calculate probabilty of Finding = T given Diseasese values
    for i = 1:length(traning_data)
       for j = 1:Nfindings
           for k = 1:Ndiseases
               if traning_data{i}(Nfindings+k) == 2 && traning_data{i}(j) == 2
                inhibit(k,j) = inhibit(k,j) + 1;
               end
           end
       end
    end
    
    inhibit = inhibit ./ sum_all; %joint probabilty distribution, marginalised to two variables - P(B_1=T, A_1=T)  *ones(size(inhibit))
    
    for l=1:Ndiseases
        inhibit(l,:) = 1 - inhibit(l,:) ./ prior(l); % p_i = P(B_1=T|A_1=T)
    end
end
