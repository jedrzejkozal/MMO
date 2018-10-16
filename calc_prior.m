function prior = calc_prior(traning_data, Nfindings, Ndiseases)
    prior = zeros(1, Ndiseases);
    %calculate probabilty, that roots are true
    for i = 1:length(traning_data)
       for j = 1:Ndiseases
          prior(j) = prior(j) + traning_data{i}(Nfindings+j)-1; 
       end
    end
    
    for k = 1:Ndiseases
        prior(k) = prior(k)/length(traning_data);
    end
end