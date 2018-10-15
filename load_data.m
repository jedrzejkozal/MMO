function data = load_data()

filename = 'diagnosis.data';
delimiterIn = ',';
raw_data = importdata(filename,delimiterIn);

number_of_instances = length(raw_data)/2-.5;
data = cell(1,number_of_instances);
data_matrix = zeros(number_of_instances, 8);

for i = 1:2:length(raw_data)-1
    index = i/2+.5;
    temperature = str2num(strcat(raw_data{i}(2),raw_data{i}(4),raw_data{i}(6),raw_data{i}(8)));
    
    if temperature > 37
       temper = 2;
    else
       temper = 1;
    end
    
    boolean = [];
    
    for j = 9:length(raw_data{i})
       if raw_data{i}(j) == 'y'
           boolean = [boolean 2];
       else
           if raw_data{i}(j) == 'n'
               boolean = [boolean 1];
           end
       end
    end
    
    data{index} = [temper boolean];
    data_matrix(index, :) = data{index};
end

end