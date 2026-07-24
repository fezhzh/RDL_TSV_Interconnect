cd(fileparts(mfilename('fullpath')));
close all
clear all

script_dir = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, "..", "..", "..");
result_root = fullfile(project_root, "model_versions", "v09_rdl_lhs_dataset_comparison", "results", "extracted_params");
model_root = fullfile(project_root, "model_versions", "v09_rdl_lhs_dataset_comparison", "models", "matlab_param_nns");
summary_file = fullfile(model_root, "matlab_training_summary.csv");

TRAIN_ONLY = true;
iterate = 1;
name_list = ["R1", "R2", "R3", "L1", "L2", "L3", "Cox", "Csi", "Rsi"];
dataset_list = ["lhs100", "lhs200", "lhs400", "lhs800", "lhs100_lhs200_lhs400_lhs800"];

devices(1).name = "TMRDL";
devices(1).model_prefix = "TMRDL_";
devices(1).input_columns = ["pitch", "l_tmrdl", "w_tmrdl", "h_tmrdl"];

devices(2).name = "BSMRDL";
devices(2).model_prefix = "BSMRDL_";
devices(2).input_columns = ["pitch", "l_bsmrdl", "w_bsmrdl", "h_bsmrdl"];

if ~exist(model_root, 'dir')
    mkdir(model_root);
end

summary_dataset = strings(0, 1);
summary_device = strings(0, 1);
summary_param = strings(0, 1);
summary_samples = [];
summary_max_error = [];
summary_average_error = [];
summary_mat_file = strings(0, 1);

for ds = 1:numel(dataset_list)
    dataset_name = dataset_list(ds);
    dataset_model_dir = fullfile(model_root, char(dataset_name));
    if ~exist(dataset_model_dir, 'dir')
        mkdir(dataset_model_dir);
    end

    for d = 1:numel(devices)
        device = devices(d);
        data_file = fullfile(result_root, char(dataset_name), strcat(device.name, "_circuit_params.csv"));
        fprintf("\nTraining MATLAB NN for %s / %s\n", dataset_name, device.name);
        fprintf("Data file: %s\n", data_file);

        if ~exist(data_file, 'file')
            error("Missing extracted parameter CSV: %s", data_file);
        end

        data_table = readtable(data_file, 'VariableNamingRule', 'preserve');
        if TRAIN_ONLY && any(strcmp(data_table.Properties.VariableNames, "split"))
            data_table = data_table(string(data_table.split) == "train", :);
        end

        input = table2array(data_table(:, cellstr(device.input_columns)));
        valid_input = all(isfinite(input), 2);
        input = input(valid_input, :);
        data_table = data_table(valid_input, :);

        for j = 1:numel(name_list)
            param_name = name_list(j);
            output_v = data_table.(char(param_name));
            valid_rows = isfinite(output_v) & output_v > 0;
            input_j = input(valid_rows, :);
            output_j = output_v(valid_rows, :);

            if isempty(input_j)
                warning("%s %s %s has no valid rows, skipped.", dataset_name, device.name, param_name);
                continue;
            end

            e3 = 1e16;
            e4 = 1e16;

            for i = 1:iterate
                net = NN_train(input_j', output_j');
                [MaxError, AverageError] = NN_error(input_j, output_j, net);
                e1 = MaxError
                e2 = AverageError

                if (e1 <= e3) && (e2 <= e4)
                    e3 = e1;
                    e4 = e2;
                    net_p = net;
                end
            end

            mat_file = fullfile(dataset_model_dir, strcat(device.model_prefix, param_name, ".mat"));
            NN_export(net_p, input_j, output_j, mat_file, param_name)
            fprintf("%s / %s %s saved to %s\n", dataset_name, device.name, param_name, mat_file);

            summary_dataset(end + 1, 1) = dataset_name;
            summary_device(end + 1, 1) = device.name;
            summary_param(end + 1, 1) = param_name;
            summary_samples(end + 1, 1) = size(input_j, 1);
            summary_max_error(end + 1, 1) = e3;
            summary_average_error(end + 1, 1) = e4;
            summary_mat_file(end + 1, 1) = string(mat_file);

            clearvars net*
        end
    end
end

summary_table = table( ...
    summary_dataset, ...
    summary_device, ...
    summary_param, ...
    summary_samples, ...
    summary_max_error, ...
    summary_average_error, ...
    summary_mat_file, ...
    'VariableNames', {'dataset', 'device', 'parameter', 'samples', 'max_relative_error', 'average_relative_error', 'mat_file'} ...
);
writetable(summary_table, summary_file);
fprintf("\nTraining summary saved to %s\n", summary_file);


%% NN train
function net=NN_train(x,t)

    trainFcn = 'trainbr';
    net = feedforwardnet([20,20],trainFcn);
    net.trainParam.epochs=2000;
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};

    net.divideFcn = 'dividerand';
    net.divideMode = 'sample';
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;

    net.performFcn = 'mse';
    net.plotFcns = {};

    [net,tr] = train(net,x,t);
    y = net(x);
    performance = perform(net,t,y)
    trainTargets = t .* tr.trainMask{1};
    valTargets = t .* tr.valMask{1};
    testTargets = t .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,y);
    valPerformance = perform(net,valTargets,y);
    testPerformance = perform(net,testTargets,y);

end


%% Network error
function [MaxError,AverageError] = NN_error(input,output_v,net)

    for i = 1:1:size(input,2)
        xmax(i) = max(input(:,i));
        xmin(i) = min(input(:,i));
        inputn(:,i) = 2*(input(:,i)-xmin(i))/(xmax(i)-xmin(i))-1;
    end

    w1 = net.iw{1,1}';
    theta1 = net.b{1}';
    w2 = net.lw{2,1}';
    theta2 = net.b{2}';
    w3 = net.lw{3,2}';
    theta3 = net.b{3}';

    outputn = (2./(1+exp(-2*((2./(1+exp(-2*(inputn*w1+theta1)))-1)*w2+theta2)))-1)*w3+theta3;

    outputmax = max(output_v);
    outputmin = min(output_v);
    output_ = (outputmin+(outputn+1).*(outputmax-outputmin)./2);

    error = abs((output_-output_v)./output_v);
    MaxError = max(abs(error));
    AverageError = sum(abs(error))/length(output_v);

end


function NN_export(net_p,input,output,matname,valname)
    psmin = min(input,[],1);
    psmax = max(input,[],1);
    outputmax = max(output);
    outputmin = min(output);

    w1 = net_p.iw{1,1}';
    theta1 = net_p.b{1}';
    w2 = net_p.lw{2,1}';
    theta2 = net_p.b{2}';
    w3 = net_p.lw{3,2}';
    theta3 = net_p.b{3}';

    matdir = fileparts(matname);
    if ~isempty(matdir) && ~exist(matdir, 'dir')
        mkdir(matdir);
    end

    save(matname,'psmax', 'psmin', 'w1', 'theta1', 'w2', 'theta2', 'w3', 'theta3', 'outputmax', 'outputmin','valname')
end
