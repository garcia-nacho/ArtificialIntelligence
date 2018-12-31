library(ggplot2)
library(h2o)

#Episode Solved
dfWeights$Completed<-"NO"
dfWeights$Completed[dfWeights$V11>=195]<-"YES"

#Linear Regression
dfWeights$RatioWN2<-dfWeights$V9/dfWeights$V10
dfWeights$LinReg<-"NO"
dfWeights$LinReg[abs(dfWeights$RatioWN2)<0.04 | abs(dfWeights$RatioWN2)>25 ]<-"YES"
  
ggplot(dfWeights)+
  geom_jitter(aes(x=V9, y=V10, colour=Completed, shape=LinReg), size=3, alpha=0.5)+
  theme_minimal()

#NNet to find new models weights
h2o.init()
dfWeights$LinReg<-NULL
dfWeights$V11<-NULL
dfWeights$RatioWN2<-NULL

dfWeights$Completed[dfWeights$Completed=="YES"]<-1
dfWeights$Completed[dfWeights$Completed=="NO"]<-0
dfWeights$Completed<-as.factor(dfWeights$Completed)

dfWeightsTrain<-dfWeights[1:900,]
dfWeightsTest<-dfWeights[901:1000,]

df.h2oTrain <- as.h2o(dfWeightsTrain)
df.h2oTest <- as.h2o(dfWeightsTest)
nfolds<-5


#Loop

for (loop in 1:10){

df.h2oWeights<-as.h2o(dfWeights)

my_NNet	<-	h2o.deeplearning(1:10, 
                        y=11,
                        stopping_metric="logloss",
                        distribution = "bernoulli",
                        loss = "CrossEntropy",
                        balance_classes = TRUE,
                        #validation_frame = df.h2oTest,
                        training_frame	=	df.h2oWeights,
                        hidden = c(24,24),
                        nfolds = nfolds,
                        fold_assignment = "Modulo",
                        keep_cross_validation_predictions = TRUE,
                        epochs = 10)


P<-h2o.predict(my_NNet, df.h2oTest)
dfWeightsTest$predicted<-as.vector(P$predict)


weights.space <- as.data.frame(matrix(runif(10000, min = -1, max = 1), ncol = 10, nrow = 1000))
weights.space.h2o<- as.h2o(weights.space)

Q<-h2o.predict(my_NNet, weights.space.h2o)
weights.space$predicted<-as.vector(Q$predict)

#Restart the gym
instance_id <- env_create(server, env_id)
Agents<-subset(weights.space, weights.space$predicted == 1)

for (k in 1:nrow(Agents)) {
  weightsT<-Agents[k,1:10]
  
  # dfGod is the dataframe where all the observation for all iterations are stored
  dfGod<-as.data.frame(t(c(1:9)))
  dfGod<-dfGod[!1,]
  
  print(paste("Round",k,sep = " "))
  for (i in 1:(iterations)) {
    dfEyes<-as.data.frame(t(rep(NA,4)))
    colnames(dfEyes)<-c("Obs1","Obs2","Obs3","Obs4")
    
    ob <- env_reset(server, instance_id)
    dfEyes[1,1:4]<-ob
    dfEyes$iteration[1]<-0
    dfEyes$Reward[1]<-NA
    dfEyes$Done[1]<-FALSE
    dfEyes$action[1]<-NA
    
    for (j in 1:max_steps) {

      Input<-dfEyes[j,1:Observations]
      
      Neuron.layer[1]<-sigmoid(sum(Input[, 1:Observations]*weightsT[1:4]), method = "tanh")  
      Neuron.layer[2]<-sigmoid(sum(Input[, 1:Observations]*weightsT[5:8]), method = "tanh")
      
      
      action<-round(sigmoid(sum(Neuron.layer*weightsT[9:10]), method = "logistic"))
      results <- env_step(server, instance_id, action, render = TRUE)
      
      dfEyes[j+1,1:4]<-results[[1]]
      dfEyes$Reward[j+1]<-results[[2]]
      dfEyes$Done[j+1]<-results[[3]]
      dfEyes$action[j+1]<-action
      dfEyes$iteration[j+1]<-j
      dfEyes$round<-i
      t.zero<-unlist(ob)
      
      if (results[["done"]]) break
    }
    dfEyes$Reward<-nrow(dfEyes)
    colnames(dfGod)<-colnames(dfEyes)
    dfGod[(nrow(dfGod)+1):(nrow(dfGod)+nrow(dfEyes)),]<-dfEyes
    
  }
  
  dfGod<-dfGod[complete.cases(dfGod),]
  Reward<-mean(unique(dfGod$Reward))
  RewardT[k]<-Reward

  print(Reward)

  
}

# Closing the environment
env_close(server,instance_id)

Agents$predicted<-RewardT
Agents$predicted[Agents$predicted >= 195]<-1
Agents$predicted[Agents$predicted != 1]<-0
Agents$predicted <- as.factor(Agents$predicted)
colnames(Agents)[11] <- "Completed"

dfWeights<-rbind(dfWeights, Agents)
}