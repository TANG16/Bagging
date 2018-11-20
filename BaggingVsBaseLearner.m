
% 36 dataset istenen veri kumesi secilir. Bu data set karistirilir. Daha
% sonra %50 Egitim, %50 Test kumesi olacak sekilde sinif bilgileri ile
% alinir. Egitim kumesi ile karar agaci modellenir. Daha sonra test kumesi
% uzerinde tahminleme yapilir. Ayni egitim kumesi uzerine Bagging
% uygulanir. Bagging de 50 adet karar agaci ile kararlar birlestirilir. Ve
% demokrasi usulu ile birlestiririlir. Bu islemeler 10 kez tekrarlanarak
% sonuclar karsilastirilir.

clc;
clear;

tumdatasets={'abalone','anneal','audiology','autos','balance-scale','breast-cancer', ...
    'breast-w','col10','colic','credit-a','credit-g','d159','diabetes','glass','heart-statlog',  ...
    'hepatitis','hypothyroid','ionosphere','iris','kr-vs-kp','labor','letter', ...
    'lymph','mushroom','primary-tumor','ringnorm','segment','sick','sonar','soybean', ...
    'splice','vehicle','vote','vowel','waveform','zoo'}; % toplam 36 dataset


[ornekler,etiketler]  =arffoku(sprintf('%s%s.arff','./36uci/',tumdatasets{6}));

% Yuzde 50 test, Yuzde 50 egitim

baseLearnerAccuracyArray=[];
emsembleLearnerAccuracyArray=[];

%  10 kez egitim kumesini karistiralim ve random secelim.
for i=1:10

shuffle = randperm(size(ornekler,1));
mixedOrnekler = ornekler(shuffle,:);
mixedEtiketler = etiketler(shuffle,end);

halfOfDataSet=round(size(ornekler,1)/2);

trainSet= mixedOrnekler(1:halfOfDataSet,1:end);
trainSetLabels=mixedEtiketler(1:halfOfDataSet,1:end);

testSet=mixedOrnekler(halfOfDataSet+1:end,1:end);
testLabels=mixedEtiketler(halfOfDataSet+1:end,1:end);

treeModel = fitctree(trainSet,trainSetLabels);

result= treeModel.predict(testSet);

% Base learner basarisi
baseLearnerAccuracy=sum(result==testLabels)/size(result,1);
baseLearnerAccuracyArray=[baseLearnerAccuracyArray baseLearnerAccuracy];

%% Bootstrap - Bagging
trainSize=size(trainSet);

% Round sayisi
roundsNumber=trainSize(1,1);

% Base learner sayisi
learnerCount=50;

 % Modellerin predictlerini saklayalim
predictResultsAcc=[];

% Egitici sayisi kadar 
for i=1:learnerCount 

% Yeni egitim ve etiketler
newRandomTrainSet=[];
newRandomTrainSetClasses=[];

     % Egitici sayisi kadar random egitim kumesi olustur, rounds
     for j=1:roundsNumber 

        % Egitim kumesinden rastgele index cek
        index = ceil(trainSize(1,1)*rand(1,1));

        % Yeni egitim ve etiketler
        newRandomTrainSet=[newRandomTrainSet ; trainSet(index,1:end)];
        newRandomTrainSetClasses=[newRandomTrainSetClasses ; trainSetLabels(index)];
     end
 
    treeModel = fitctree(newRandomTrainSet,newRandomTrainSetClasses);  

    result= treeModel.predict(testSet);
    
    % Modellerin predictlerini saklayalim
    predictResultsAcc=[predictResultsAcc result];
    
end

ensembleResult=[];

 % Demokrasi en fazla hangi sinif ise onu karar olarak alinir
 for k=1:size (predictResultsAcc ,1) 
     % satirda en cok soyleneni bulalim
     [ii,jj,kk]=unique(predictResultsAcc(k,1:end).','rows','stable');
     out=[ii,accumarray(kk,1)];
     
     [B,I] = sort(out(1:end,2),'descend');
     
      maxClass=  out(I(1,1));
     
     ensembleResult=[ensembleResult ; maxClass];
 end
 
  % Ensemble basarisi
  ensembleAccuracy=sum(ensembleResult==testLabels)/roundsNumber; 
  emsembleLearnerAccuracyArray =[emsembleLearnerAccuracyArray ; ensembleAccuracy];
end

%% Bootstrap - Bagging vs Base Learner sonuclar
array = [baseLearnerAccuracyArray' emsembleLearnerAccuracyArray]
bar(array)




