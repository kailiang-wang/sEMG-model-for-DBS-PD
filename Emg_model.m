%% Emg_model,codes by kailiang wang, 2019.12, Beijing neurosurgical institute

function  [Tremor_predict,R_model]= Emg_model(Emg_feature,UPDRS_t)
no_sub=size(Emg_feature,1);
Tremor_predict=zeros(no_sub,1);

  %% leaving one out
for leftout = 1:no_sub
Real_Data=Emg_feature;
Real_Data(:,end+1)=UPDRS_t;

trainingData=Real_Data;
trainingData(leftout,:)=[];

[trainedModel, validationRMSE] = trainRegressionModel(trainingData);
TestData=Real_Data(leftout,1:end-1)
True_predict = trainedModel.predictFcn(TestData)

Tremor_predict(leftout,1)=True_predict;
end 
[R_model,P_model]=corr(UPDRS_t,Tremor_predict);
end