
cd(fileparts(mfilename('fullpath')));
close all
clear all


data1 = xlsread("../../data/tables/RDL_Bottom_TD_4_.csv");
data = data1;%+data2;
% fit_list = [6:14];%
fit_list = [6:14];%
name_list = ["R1","R2","R3","L1", "L2", "L3", "Cox", "Csi", "Rsi"];%
iterate = 1;%
MaxErrorLists=[];
AverageErrorLists=[];




if ~isempty(data)
    input=[];
    output_v=[];
    for j = 1:1:size(fit_list,2)

            
        input = data(:,[1:5]);
        output_v = data(:,fit_list(j));
        output_vref = data(:,fit_list(j));
    
        MaxErrorList = [];
        AverageErrorList = [];
        e3 = 1e16;
        e4 = 1e16;
        for i =1:1:iterate
            net=NN_train(input',output_v');



            [MaxError,AverageError]=NN_error(input,output_v,net);
            e1 = MaxError
            e2 = AverageError
        
            if (e1<=e3) && (e2<=e4)
                e3 = e1;
                e4 = e2;
                net_p = net;
                % [in,index]=sort(error);
            end


            MaxErrorList = [MaxErrorList,MaxError];
            AverageErrorList = [AverageErrorList,AverageError];
        end
        MaxErrorLists=[MaxErrorLists;[j,MaxErrorList]];
        AverageErrorLists=[AverageErrorLists;[j,AverageErrorList]];
        NN_export(net_p,input,output_v,strcat('../../data/matlab_models/RDL_TSV_mat4/RDL_Bottom_',name_list(j),'.mat'),name_list(j))
        clearvars net*
    end

end







%% NN train
function net=NN_train(x,t)

    % 'trainlm' is usually fastest; 'trainbr' takes longer but may be better for challenging problems;'trainscg' uses less memory. Suitable in low memory situations.
    trainFcn = 'trainbr';  % Levenberg-Marquardt backpropagation.
    net = feedforwardnet([20,20],trainFcn);
    net.trainParam.epochs=2000;
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};

    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;

    net.performFcn = 'mse';  % Mean Squared Error
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
        'plotregression', 'plotfit'};

    % Train the Network
    [net,tr] = train(net,x,t);
    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    performance = perform(net,t,y)
    % Recalculate Training, Validation and Test Performance
    trainTargets = t .* tr.trainMask{1};
    valTargets = t .* tr.valMask{1};
    testTargets = t .* tr.testMask{1};
    trainPerformance = perform(net,trainTargets,y);
    valPerformance = perform(net,valTargets,y);
    testPerformance = perform(net,testTargets,y);


    if (false)
        % Generate MATLAB function for neural network for application
        % deployment in MATLAB scripts or with MATLAB Compiler and Builder
        % tools, or simply to examine the calculations your trained neural
        % network performs.
        genFunction(net,'myNeuralNetworkFunction');
        y = myNeuralNetworkFunction(x);
    end
    if (false)
        % Generate a matrix-only MATLAB function for neural network code
        % generation with MATLAB Coder tools.
        genFunction(net,'myNeuralNetworkFunction','MatrixOnly','yes');
        y = myNeuralNetworkFunction(x);
    end
    if (false)
        % Generate a Simulink diagram for simulation or deployment with.
        % Simulink Coder tools.
        gensim(net);
    end

end

    
    
    
%% Network error
function [MaxError,AverageError] = NN_error(input,output_v,net)



    for i = 1:1:size(input,2)
        xmax(i) = max(input(:,i));
        xmin(i) = min(input(:,i));
        inputn(:,i) = 2*(input(:,i)-xmin(i))/(xmax(i)-xmin(i))-1;
    end


%wight and bias of the trained network
    w1 = net.iw{1,1}';
    theta1 = net.b{1}';
    w2 = net.lw{2,1}';
    theta2 = net.b{2}';
    w3 = net.lw{3,2}';
    theta3 = net.b{3}';


    % outputn = (2./(1+exp(-2*(inputn*w3+theta3)))-1)*w4+theta4;
    % o1 = (2./(1+exp(-2*(inputn*w3+theta3)))-1);
    % o2 = (2./(1+exp(-2*((2./(1+exp(-2*(inputn*w3+theta3)))-1)*w4+theta4)))-1);

    outputn = (2./(1+exp(-2*((2./(1+exp(-2*(inputn*w1+theta1)))-1)*w2+theta2)))-1)*w3+theta3;

    outputmax = max(output_v);
    outputmin = min(output_v);
    output_ = (outputmin+(outputn+1).*(outputmax-outputmin)./2);

    error = abs((output_-output_v)./output_v);
    MaxError = max(abs(error));
    AverageError = sum(abs(error))/length(output_v);



    % outputmax = max(output_v);
    % outputmin = min(output_v);
    % output_ = (outputmin+(outputn+1).*(outputmax-outputmin)./2);

    % error = abs((output_-output_v)./output_v)*100;
    % error_max = max(abs(error))
    % av_error = sum(abs(error))/length(data)

    % output_ = exp(output_);
    % error = abs((output_-data(:,8))./data(:,8))*100;
    % error_max = max(abs(error))
    % av_error = sum(abs(error))/length(data)

    % parameters = [];
    % parameters=[reshape(xmax,[1,numel(xmax)]), reshape(xmin,[1,numel(xmin)]), reshape(w1,[1,numel(w1)]),...
    %     reshape(theta1,[1,numel(theta1)]), reshape(w2,[1,numel(w2)]), reshape(theta2,[1,numel(theta2)]), ...
    %     reshape(w3,[1,numel(w3)]), reshape(theta3,[1,numel(theta3)]), outputmax, outputmin];
    % save('D:\parameters_matlab.txt','parameters','-ascii')




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

