appliance_list = char('dishwasher1', 'clotheswasher1', 'drye1', 'oven1', 'microwave1', 'refrigerator1', 'furnace1', 'bathroom1', 'bedroom1', 'diningroom1', 'car1', 'heater1', 'livingroom1', 'poolpump1');
pearson_coeffs = zeros(1, length(appliance_list));

shift_back_interval = 1;

for appliance=1:length(appliance_list)
    
    current_directory = strcat(pwd, '/');
    
    house_id = '4874';
    
    filename = strcat(house_id, '/', house_id, '_power_values_');
    
    filename = strcat(filename, appliance_list(appliance,:), '.csv');
    
    if exist([current_directory filename], 'file') ~= 2
        continue 
    end

    power = load(filename);
    power = power(:,3);
    
    if min(power) == 0.0 && max(power) == 0.0
        continue
    end

    one_week = 672;
    one_day = 96;

    order = 1;
    number_of_weeks = 3;
    number_of_days = 7;

    training_window = number_of_weeks*one_week;

    input = zeros(one_day, training_window);

    for i=1:one_day
       input(i,:) = power(i:training_window-1+i);
    end

    observation = power(one_day+1:one_day+training_window);
    input = input';

    weigths = TeslaTrain(input, observation, order);

    prediction_window = one_day*number_of_days;
    
    if min(power(1:one_week+prediction_window)) == 0.0 && max(power(1:one_week+prediction_window)) == 0.0
        continue
    end

    prediction_input = zeros(prediction_window, one_day);

    for i=1:prediction_window
        prediction_input(i,:) = power(training_window-one_day+i:training_window-1+i); 
    end

    results = TeslaPredict(weigths, order, prediction_input);

    error = zeros(prediction_window,1);
    deviation = zeros(prediction_window,1);

    for i=1:prediction_window
        deviation(i) = abs(results(i) - power(training_window+i-shift_back_interval));
        error(i) = deviation(i)/abs(results(i))*100;
    end
    
    observed_values = power(training_window+1-shift_back_interval:training_window+prediction_window-shift_back_interval);
    
    figure()
    plot(1:prediction_window, abs(results), '-x', 1:prediction_window, observed_values, '-o');
    
    title([house_id ' ' appliance_list(appliance,:)])
    xlabel('Time intervals')
    ylabel('Power(kW)')

    % plot(1:prediction_window, error);

    C=cov(results,observed_values);
    pearson_coeffs(appliance)=C(2)/(std(results)*std(observed_values));


end