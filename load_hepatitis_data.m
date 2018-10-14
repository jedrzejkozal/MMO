% function loads hepatitis data
% -9 is equivalent of absence of parameter
function hepatitis_data = load_hepatitis_data()
filename = 'hepatitis.data';
delimiterIn = ',';
headerlinesIn = 0;
raw_data = importdata(filename,delimiterIn,headerlinesIn);
hepatitis_data = cell(1,length(raw_data));
for i = 1:length(raw_data)
    hepatitis_data(i) = {raw_data(i,:)};
end

end