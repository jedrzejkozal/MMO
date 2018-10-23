function data = generate_data()
    number_of_instances = 120;
    data = cell(1, number_of_instances);
    
    d1 = rand(1, number_of_instances) + 100;%normrnd(100,1,1,number_of_instances);
    d2 = rand(1, number_of_instances) + 300;%normrnd(120,5,1,number_of_instances);
    
    s1 = (2*d1 +3);
    s2 = log(d1);
    s3 = 3*d1.*d1 + d1 - 3;
    
    s4 = 4*d2 + 7;
    s5 = 10*log(d2);
    s6 = 4*d2.*d2 - 2*d2 + 4;
%     
%     r1 = round(rand(1,number_of_instances))+1;
%     r2 = round(rand(1,number_of_instances))+1;
%     
%     d1 = (d1 > 100 & d1 < 120) + 1;
%     d2 = (d2 > 120) + 1;
%     
%     s1 = (s1 > 203 & s1 < 487) + 1;
%     s2 = (s2 > 4.6 & s2 < 47.9) + 1;
%     s3 = (s3 > 29997 & s3 < 57364) + 1;
%     
%     s4 = (s4 > 487) + 1;
%     s5 = (s5 > 47.9) + 1;
%     s6 = (s6 > 57364) + 1;
%     
%     r1 = round(rand(1,number_of_instances))+1;
%     r2 = round(rand(1,number_of_instances))+1;
    
    for i = 1:number_of_instances
        data{i} = int16([s1(i) s2(i) s3(i) s4(i) s5(i) s6(i) d1(i) d2(i)]);
    end
end