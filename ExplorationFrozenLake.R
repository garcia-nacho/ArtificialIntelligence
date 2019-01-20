#Nacho Garcia 2019
#garcia.nacho@gmail.com

library(gym)
library(ggplot2)

server <- create_GymClient("http://127.0.0.1:5000")

# Create environment
env_id <- "FrozenLake-v0"
instance_id <- env_create(server, env_id)

#Variables
iterations <- 1500
  
  
  for (i in 1:(iterations)) {
    dfGod <- as.data.frame(t(rep(NA, 7)))
    colnames(dfGod) <- c("Position", "Action", "Reward", "Done", "Fail","Step","Round")
    ob <- env_reset(server, instance_id)
    dfGod$Position[1] <- ob
    
    stop <- 0
    j<-1
    while(stop == 0) {
      
      action <- env_action_space_sample(server, instance_id)
      dfGod$Action[j] <- action
      
      results <- env_step(server, instance_id, action, render = TRUE)
      dfGod$Done[j] <- results$done
      dfGod$Reward[j] <- results$reward
      dfGod$Step[j] <- j
      dfGod$Round[j] <- i
      
      dfGod[j+1,]<-NA
      dfGod$Position[j+1]<-results$observation
      
      if (results$done==TRUE) print(paste("Stopping Iteration", i, sep = " "))
      if (results$done==TRUE) stop <- 1
      j<-j+1
      }
    
    if(i==1) dfGod2<-dfGod 
    if(i>1) dfGod2<-rbind(dfGod2,dfGod)
  }

env_close(server,instance_id)

dfGod2$Fail <- 0
dfGod2$Fail[dfGod2$Reward==0 & dfGod2$Done==1]<-1

dfGod2<-dfGod2[complete.cases(dfGod2),]

#One-hot Encoding of States and Actions
OHE.Position <- as.data.frame(matrix(data=0, nrow = nrow(dfGod2), ncol = 15 ))
for (i in 1:nrow(dfGod2)) {
  position<-as.numeric(dfGod2$Position[i])
  OHE.Position[i,position+1]<-1
}

OHE.Action <- as.data.frame(matrix(data=0, nrow = nrow(dfGod2), ncol = 4 ))
colnames(OHE.Action)<-c("A0","A1","A2","A3")
for (i in 1:nrow(dfGod2)) {
  position<-as.numeric(dfGod2$Action[i])
  OHE.Action[i,position+1]<-1
}

dfGod<-cbind(OHE.Position,OHE.Action)
dfGod[,20:24]<-dfGod2[,3:7]

 
## Training Against Failures -----
h2o.init()

dfGod$Fail<-as.factor(dfGod$Fail)
df.h2oTrain <- as.h2o(dfGod)
nfolds <- 5

        FrozenLake_NNet	<-	h2o.deeplearning(x= 1:19,
                            y= 22,
                            stopping_metric="logloss",
                            distribution = "bernoulli",
                            loss = "CrossEntropy",
                            balance_classes = TRUE,
                            #validation_frame = df.h2oTest,
                            training_frame	=	df.h2oTrain,
                            hidden = c(24,24),
                            nfolds = nfolds,
                            fold_assignment = "Modulo",
                            keep_cross_validation_predictions = TRUE,
                            epochs = 20)

h2o.auc(FrozenLake_NNet)

#Precompute all actions for each position
PositionSpace<- as.data.frame(c(1,2,3,4,5,7,9,10,11,14,15))
colnames(PositionSpace)<-"Position"
OHE.Position <- as.data.frame(matrix(data=0, nrow = nrow(PositionSpace), ncol = 15 ))
for (i in 1:nrow(PositionSpace)) {
  position<-as.numeric(PositionSpace$Position[i])
  OHE.Position[i,position]<-1
}

OHE.Position$A0<-1
OHE.Position$A1<-0
OHE.Position$A2<-0
OHE.Position$A3<-0

OHE.PositionA1<-OHE.Position
OHE.PositionA1$A0<-0
OHE.PositionA1$A1<-1

OHE.PositionA2<-OHE.Position
OHE.PositionA2$A0<-0
OHE.PositionA2$A2<-1

OHE.PositionA3<-OHE.Position
OHE.PositionA3$A0<-0
OHE.PositionA3$A3<-1

SpacePosAct <- rbind(OHE.Position, OHE.PositionA1, OHE.PositionA2, OHE.PositionA3)
SpacePosAct.h2o <- as.h2o(SpacePosAct)

prediction<-h2o.predict(FrozenLake_NNet, SpacePosAct.h2o)
SpacePosAct$Prediciton <- as.vector(prediction$predict)

#Restart gym ----------
instance_id <- env_create(server, env_id)

for (i in 1:(iterations)) {
  dfGodN <- as.data.frame(t(rep(NA, 7)))
  colnames(dfGodN) <- c("Position", "Action", "Reward", "Done", "Fail","Step","Round")
  ob <- env_reset(server, instance_id)
  dfGodN$Position[1] <- ob
  
  stop <- 0
  j<-1
  while(stop == 0) {
    
    #Check for non faulty actions  
      PosAct<-subset(SpacePosAct, SpacePosAct[,dfGodN$Position[j]+1]==1 & SpacePosAct$Prediciton==0)
    #Random selection of one non faulty action  
      RndAct<-round(runif(1, min = 1, max = nrow(PosAct)))
      PosAct<-as.vector(PosAct[RndAct,16:19])
      action <- which(PosAct==1) -1 
    
    
    dfGodN$Action[j] <- action
    
    results <- env_step(server, instance_id, action, render = TRUE)
    dfGodN$Done[j] <- results$done
    dfGodN$Reward[j] <- results$reward
    dfGodN$Step[j] <- j
    dfGodN$Round[j] <- i
    
    dfGodN[j+1,]<-NA
    dfGodN$Position[j+1]<-results$observation
    
    if (results$done==TRUE) print(paste("Improved Model Iteration:", i, sep = " "))
    if (results$done==TRUE) stop <- 1
    j<-j+1
  }
  
  if(i==1) dfGodN2<-dfGodN 
  if(i>1) dfGodN2<-rbind(dfGodN2,dfGodN)
}

env_close(server,instance_id)

dfGodN2$Fail <- 0
dfGodN2$Fail[dfGodN2$Reward==0 & dfGodN2$Done==1]<-1

dfGodN2<-dfGodN2[complete.cases(dfGodN2),]


#One-hot Encoding of States and Actions
OHE.Position <- as.data.frame(matrix(data=0, nrow = nrow(dfGodN2), ncol = 15 ))
for (i in 1:nrow(dfGodN2)) {
  position<-as.numeric(dfGodN2$Position[i])
  OHE.Position[i,position+1]<-1
}

OHE.Action <- as.data.frame(matrix(data=0, nrow = nrow(dfGodN2), ncol = 4 ))
colnames(OHE.Action)<-c("A0","A1","A2","A3")
for (i in 1:nrow(dfGodN2)) {
  position<-as.numeric(dfGodN2$Action[i])
  OHE.Action[i,position+1]<-1
}

dfGodN<-cbind(OHE.Position,OHE.Action)
dfGodN[,20:24]<-dfGodN2[,3:7]


#Performance check
randomN<-rep(NA,1500)
improvedN<-rep(NA,1500)
for (i in 1:iterations){
  random<-subset(dfGod2, dfGod2$Round==i)
  randomN[i]<-max(random$Step)
  improved<-subset(dfGodN2,dfGodN2$Round==i)
  improvedN[i]<-max(improved$Step)
}


#Model x=observation y=action 


#Restart the gym for 3rd time

instance_id <- env_create(server, env_id)

for (i in 1:(iterations)) {
  dfGodN <- as.data.frame(t(rep(NA, 7)))
  colnames(dfGodN) <- c("Position", "Action", "Reward", "Done", "Fail","Step","Round")
  ob <- env_reset(server, instance_id)
  dfGodN$Position[1] <- ob
  
  stop <- 0
  j<-1
  while(stop == 0) {
    
    #Check for predicted actions
    

    action <- which(PosAct==1) -1 
    
    
    dfGodN$Action[j] <- action
    
    results <- env_step(server, instance_id, action, render = TRUE)
    dfGodN$Done[j] <- results$done
    dfGodN$Reward[j] <- results$reward
    dfGodN$Step[j] <- j
    dfGodN$Round[j] <- i
    
    dfGodN[j+1,]<-NA
    dfGodN$Position[j+1]<-results$observation
    
    if (results$done==TRUE) print(paste("Improved Model Iteration:", i, sep = " "))
    if (results$done==TRUE) stop <- 1
    j<-j+1
  }
  
  if(i==1) dfGodN2<-dfGodN 
  if(i>1) dfGodN2<-rbind(dfGodN2,dfGodN)
}

env_close(server,instance_id)

dfGodN2$Fail <- 0
dfGodN2$Fail[dfGodN2$Reward==0 & dfGodN2$Done==1]<-1

dfGodN2<-dfGodN2[complete.cases(dfGodN2),]
