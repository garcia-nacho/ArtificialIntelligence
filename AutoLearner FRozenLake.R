library(gym)
library(ggplot2)

server <- create_GymClient("http://127.0.0.1:5000")


# Create environment
env_id <- "FrozenLake-v0"
iterations <- 1000
Suc <- 0
Suc.rate <- rep(NA, iterations)

#Learning Curve

learn.curve <- function(x, min, slope, L50){
  min+((1-min)/(1+exp(-slope*(L50-x))))
}

min<-0.05
slope<-0.05
L50<-100

 a <- learn.curve(1:250,min,slope,L50)
 plot(a, type = "l", ylim = c(0,1))


Curiosity <- 0.1

#Start gym ----------
instance_id <- env_create(server, env_id)

#Growing a tree
SpaceTree<-expand.grid(c(0:16),c(0:3),c(0,1))
colnames(SpaceTree)<-c("Position","Action","Fail")
SpaceTree$N<-0
SpaceTree$Prob<-0
SpaceTree$Reward<-0
SpaceTree$Length<-0

for (i in 1:(iterations)) {
  dfGodN <- as.data.frame(t(rep(NA, 7)))
  colnames(dfGodN) <- c("Position", "Action", "Reward", "Done", "Fail","Step","Round")
  ob <- env_reset(server, instance_id)
  dfGodN$Position[1] <- ob
  
  stop <- 0
  j<-1
  while(stop == 0) {
    
    PosAct<-subset(SpaceTree, SpaceTree$Position==dfGodN$Position[j] & SpaceTree$Fail==1)
    
#Policy I mininal fail> reward > Length
    
    # action <- subset(PosAct,PosAct$Prob == min(PosAct$Prob))
    # action <- subset(action,action$Reward == max(action$Reward))
    # action <- subset(action,action$Length == min(action$Length))
    # action<-action$Action
    
#Policy II mininal  reward > Length>fail    
    
    action <- subset(PosAct,PosAct$Reward == max(PosAct$Reward))
    action <- subset(action,action$Length == min(action$Length))
    action <- subset(action,action$Prob == min(action$Prob))
    action<-action$Action
    
    
  
      if (length(action) > 1) action<- action[round(runif(1,min = 1, max = length(action)))]
    experienceN <- PosAct$N[PosAct$Action==action]
    
    Curiosity <- learn.curve(experienceN,min,slope,L50)
    
    if(Curiosity > runif(1,min =0 , max= 1) ){
      action <- env_action_space_sample(server, instance_id)
    }
    
    dfGodN$Action[j] <- action
    
    results <- env_step(server, instance_id, action, render = TRUE)
    dfGodN$Done[j] <- results$done
    dfGodN$Reward[j] <- results$reward
    dfGodN$Step[j] <- j
    dfGodN$Round[j] <- i
    
    dfGodN[j+1,]<-NA
    dfGodN$Position[j+1]<-results$observation

    if (results$done==TRUE) stop <- 1
    j<-j+1
  }
  
  dfGodN$RewardEp <- 0
  if (max(dfGodN$Reward, na.rm = TRUE)==1) dfGodN$RewardEp <-1
  
  dfGodN$Fail <- 0
  dfGodN$Fail[dfGodN$Reward==0 & dfGodN$Done==1]<-1
  dfGodN$StepMax <- max(dfGodN$Step, na.rm = TRUE)
  
  if(i==1) dfGodN2<-dfGodN 
  if(i>1) dfGodN2<-rbind(dfGodN2,dfGodN)
  
  if (max(dfGodN$Reward, na.rm = TRUE)==1) Suc<- Suc+1 
  Suc.rate[i] <- Suc/i
  
  if (max(dfGodN$Reward, na.rm = TRUE)==1) print(paste("Iteration:",i, " OK", sep = "" ))
  if (max(dfGodN$Reward, na.rm = TRUE)==0) print(paste("Iteration:",i, " FAIL", sep = "" ))
  

  dfGodN2<-dfGodN2[complete.cases(dfGodN2),]
  
 #Transfer data to the tree
  for (i in 1:nrow(SpaceTree)) {
    SpaceTree$Prob[i] <- length(dfGodN2$Position[dfGodN2$Position==SpaceTree$Position[i] &
                                                  dfGodN2$Action==SpaceTree$Action[i] & 
                                                  dfGodN2$Fail==1])/
                                length(dfGodN2$Position[dfGodN2$Position==SpaceTree$Position[i] &
                                dfGodN2$Action==SpaceTree$Action[i]])
    
    SpaceTree$N[i] <-length(dfGodN2$Position[dfGodN2$Position==SpaceTree$Position[i] &
                                              dfGodN2$Action==SpaceTree$Action[i] ])
    
    SpaceTree$Reward[i] <- length(dfGodN2$Position[dfGodN2$Position==SpaceTree$Position[i] &
                                                    dfGodN2$Action==SpaceTree$Action[i] & 
                                                    dfGodN2$RewardEp==1])/
                                                    length(dfGodN2$Position[dfGodN2$Position==SpaceTree$Position[i] &
                                                    dfGodN2$Action==SpaceTree$Action[i]])
    
    #Check length 
    SpaceTree$Length[i]<-mean(dfGodN2$StepMax[dfGodN2$Position==SpaceTree$Position[i] &
                                               dfGodN2$Action==SpaceTree$Action[i]&
                                               dfGodN2$RewardEp==1])
    
  }
  SpaceTree$Prob[is.nan(SpaceTree$Prob)] <- 0
  SpaceTree$Reward[is.nan(SpaceTree$Reward)] <- 0
  SpaceTree$Length[is.na(SpaceTree$Length)] <- iterations

  }
plot(Suc.rate, type = "l")
env_close(server,instance_id)





