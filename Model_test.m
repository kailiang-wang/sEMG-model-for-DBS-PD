%% codes by kailiang wang, 2019.12, Beijing neurosurgical institute
%% original data
load data;
Emg_feature=[MNP,PSD,wRMS];
UPDRS_t=updrs_Tremor;
%% training model
[Tremor_predict,R_model]= Emg_model(Emg_feature,UPDRS_t);

True_Tremor_predict=Tremor_predict;

%% Permutation test
no_sub=35;
Ture_Prediction_r=R_model;
no_iteration = 1000;
prediction_r= zeros(no_iteration,1);
prediction_r(1) = Ture_Prediction_r;
h = waitbar(0,'please wait..');
for it=2:no_iteration
    waitbar(it/no_iteration,h,[num2str(it),'/',num2str(no_iteration)]);
    fprintf('\n Performing iteration %d out of %d',it, no_iteration);
    
    new_tremor =UPDRS_t(randperm(no_sub));
    
 
  [Tremor_predict_rand,R_model_rand]=Emg_model(Emg_feature,new_tremor);
   
  prediction_r(it)=R_model_rand;
    
end
close(h);

pval_pred = mean(prediction_r >= Ture_Prediction_r);

figure(1); plot(UPDRS_t, True_Tremor_predict,'r*');lsline;

