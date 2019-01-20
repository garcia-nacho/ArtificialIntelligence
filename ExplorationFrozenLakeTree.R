library(gym)
library(ggplot2)

server <- create_GymClient("http://127.0.0.1:5000")

# Create environment
env_id <- "FrozenLake-v0"
instance_id <- env_create(server, env_id)

#Run the environment
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

#Growing a tree
SpaceTree<-expand.grid(c(0:16),c(0:3),c(0,1))
colnames(SpaceTree)<-c("Position","Action","Fail")
SpaceTree$N<-0
SpaceTree$Prob<-0
SpaceTree$Reward<-0
SpaceTree$Length<-0
dfGod2$RewardEp<-0
dfGod2$StepMax<-0
RewardEp <- unique(dfGod2$Round[dfGod2$Reward==1])
dfGod2$RewardEp[dfGod2$Round %in% RewardEp]<-1

for (i in 1:iterations) {
  dfGod2$StepMax[dfGod2$Round==i]<-max(dfGod2$Step[dfGod2$Round==i])
}

#Improved tree

for (i in 1:nrow(SpaceTree)) {
  SpaceTree$Prob[i] <- length(dfGod2$Position[dfGod2$Position==SpaceTree$Position[i] &
                                                dfGod2$Action==SpaceTree$Action[i] & 
                                                dfGod2$Fail==SpaceTree$Fail[i]])/
                        length(dfGod2$Position[dfGod2$Position==SpaceTree$Position[i] &
                                                dfGod2$Action==SpaceTree$Action[i]])
  
  SpaceTree$N[i] <-length(dfGod2$Position[dfGod2$Position==SpaceTree$Position[i] &
                                               dfGod2$Action==SpaceTree$Action[i] & 
                                               dfGod2$Fail==SpaceTree$Fail[i]])

  SpaceTree$Reward[i] <- length(dfGod2$Position[dfGod2$Position==SpaceTree$Position[i] &
                                               dfGod2$Action==SpaceTree$Action[i] & 
                                                 dfGod2$RewardEp==1])/
                      length(dfGod2$Position[dfGod2$Position==SpaceTree$Position[i] &
                                              dfGod2$Action==SpaceTree$Action[i]])
  
  SpaceTree$Length[i]<-mean(dfGod2$StepMax[dfGod2$Position==SpaceTree$Position[i] &
                                             dfGod2$Action==SpaceTree$Action[i]&
                                             dfGod2$RewardEp==1])
  
}


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
    #Check all non faulty probabilities and select the less "dangerous option"  
      
       PosAct<-subset(SpaceTree, SpaceTree$Position==dfGodN$Position[j] & SpaceTree$Fail==0)
       action <- subset(PosAct,PosAct$Length == min(PosAct$Length))
       #action <- action$Action[action$Length == min(action$Length)]
       action<-action$Action
       if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]
       
     
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


#Performance check
randomN<-rep(NA,1500)
improvedN<-rep(NA,1180)
for (i in 1:iterations){
  random<-subset(dfGod2, dfGod2$Round==i)
  randomN[i]<-max(random$Step)
  improved<-subset(dfGodN2,dfGodN2$Round==i)
  improvedN[i]<-max(improved$Step)
}


# Curiosity/Security agent ---------------
CuriosityRate<-1

instance_id <- env_create(server, env_id)

#Growing a tree
SpaceTree<-expand.grid(c(0:16),c(0:3),c(0,1))
colnames(SpaceTree)<-c("Position","Action","Fail")
SpaceTree$N<-0
SpaceTree$Prob<-0
SpaceTree$Reward<-0


for (i in 1:(iterations)) {
  Curiosity<-5
  
  
  dfGodN <- as.data.frame(t(rep(NA, 7)))
  colnames(dfGodN) <- c("Position", "Action", "Reward", "Done", "Fail","Step","Round")
  ob <- env_reset(server, instance_id)
  dfGodN$Position[1] <- ob
  
  stop <- 0
  j<-1
  while(stop == 0) {
    
    #Check for non faulty actions  
    #Check all non faulty probabilities and select the less "dangerous option"  
    
    PosAct<-subset(SpaceTree, SpaceTree$Position==dfGodN$Position[j] & SpaceTree$Fail==1)
    
    action <- subset(PosAct,PosAct$Reward == max(PosAct$Reward))
    action <- action$Action[action$Prob == min(action$Prob)]
    if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]
    
    #Curiosity implementation
    if(runif(1,min=1,max = round(mean(PosAct$N)) +1) < Curiosity) action <- env_action_space_sample(server, instance_id)
    
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
  
  dfGodN$RewardEp<-0
  if (max(dfGodN$Reward==1)) dfGodN$RewardEp<-1

  if(i==1) dfGodN2<-dfGodN 
  if(i>1) dfGodN2<-rbind(dfGodN2,dfGodN)
  
  #Tree creation
  
  for (i in 1:nrow(SpaceTree)) {
    SpaceTree$Prob[i] <- length(dfGodN2$Position[dfGodN2$Position==SpaceTree$Position[i] &
                                                  dfGodN2$Action==SpaceTree$Action[i] & 
                                                  dfGodN2$Fail==SpaceTree$Fail[i]])/
      length(dfGodN2$Position[dfGodN2$Position==SpaceTree$Position[i] &
                               dfGodN2$Action==SpaceTree$Action[i]])
    
    SpaceTree$N[i] <-length(dfGodN2$Position[dfGodN2$Position==SpaceTree$Position[i] &
                                              dfGodN2$Action==SpaceTree$Action[i] & 
                                              dfGod2N$Fail==SpaceTree$Fail[i]])
    
    #Improved tree
    
    SpaceTree$Reward[i] <- length(dfGodN2$Position[dfGodN2$Position==SpaceTree$Position[i] &
                                                    dfGodN2$Action==SpaceTree$Action[i] & 
                                                    dfGodN2$RewardEp==1])/
      length(dfGodN2$Position[dfGodN2$Position==SpaceTree$Position[i] &
                               dfGodN2$Action==SpaceTree$Action[i]])   
    
  }
  
  
  
}

env_close(server,instance_id)

