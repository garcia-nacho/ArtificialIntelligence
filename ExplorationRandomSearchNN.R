library(gym)
library(sigmoid)

server <- create_GymClient("http://127.0.0.1:5000")

# Create environment
env_id <- "CartPole-v0"

instance_id <- env_create(server, env_id)

iterations <- 5
reward <- 0
done <- FALSE
rounds<-50
Neurons <- 2
Observations <- 4
Neuron.layer<-rep(NA,Neurons)
max_steps<-200

for (k in 1:(rounds)) {

  # dfGod is the dataframe where all the observation for all iterations are stored
  dfGod<-as.data.frame(t(c(1:9)))
  dfGod<-dfGod[!1,]
  
  # Random initialization of weights 
  weightsN1 <-  matrix(data = runif(((Observations)*Neurons), min = -1, max = 1), ncol = Observations, nrow = Neurons)
  weightsN2 <- rep(runif(Neurons,min = -1, max = 1), Neurons)

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
    
    for (h in 1:Neurons) {
      
      Neuron.layer[h]<-sigmoid(sum(Input[, 1:Observations]*weightsN1[h,]), method = "tanh")  
      
    }
    
    action<-round(sigmoid(sum(Neuron.layer*weightsN2), method = "logistic"))
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
print(Reward)
# If the mean of the reward is higher than 195 it is solved
if (Reward>=195) break

}

if (Reward>=195) print(paste("Solved after:", k, "rounds", sep = " " ))

# Closing the environment
env_close(server,instance_id)



