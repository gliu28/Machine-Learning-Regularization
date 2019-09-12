install.packages(c("tidyverse","e1071", "regclass", "caret", "glmnet", "pROC", "randomForest","xgboost", "Matrix", "shiny", "shinydashboard"))

library(shiny)
library(shinydashboard)

ui <- dashboardPage(
  dashboardHeader(title = "REGULARIZATION"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("About", tabName = "about", icon = icon("list-alt")),
      menuItem("Data", tabName = "data", icon = icon("database")),
      menuItem("Model", tabName = "model", icon = icon("cog")),
      menuItem("Logistics Regreesion", tabName = "Logistics_Regreesion", icon = icon("bar-chart-o")),
      menuItem("Ridge Regreesion", tabName = "Ridge_Regreesion", icon = icon("bar-chart-o")),
      menuItem("Lasso Regreesion", tabName = "Lasso_Regreesion", icon = icon("bar-chart-o")),
      menuItem("Elastic Net Regreesion", tabName = "Elastic_Net_Regreesion", icon = icon("bar-chart-o"))
    )
  ),

  dashboardBody(
    tabItems(
      
      # First tab content
      tabItem(tabName = "about", 
              strong("Data Driscription:"),
              br(),
              br(),
              p("(1) Data LAUNCH contains the profit of newly released products over the first few months of their release."),
              p("(2) intall.Packages(c('regclass', 'caret', 'glmnet', 'pROC', 'randomForest','xgboost', 'Matrix')."),
              p("(3) Profit which is greater than 4.5 is definded as 'Yes' and others are 'no'."),
              p("(4) Remove all columns that contain a single value."),
              p("(5) Just keep top 5 important variables based on randomforest method."),
              p("(6) Convert categorical variables into indicator variables."),
              p("(7) We want to penalty coefficient, convert the coefficient to be consistent as mean=0, sd=1."),
              p("(8) Split data into train and test as 75% and 25%."),
              p("(9) Cross-validation method is applied to estimate lambda.")
              ),
      # Second tab content
      tabItem(tabName = "data",
              dataTableOutput("Mydata")),
      
      # Third tab content
      tabItem(tabName = "model", h2("Statistics Models:"),
              
              br(),
              htmlOutput("LogisticsModel"),
              br(),
              
              htmlOutput("RidgeModel"),
              br(),
              
              htmlOutput("LassoModel"),
              br(),
            
              htmlOutput("ENModel"),
              br(),
              
              htmlOutput("Note")
              
      ),
      
      # Fourth tab content
      tabItem(tabName = "Logistics_Regreesion", h2("Logistics Regreesion"),
              
              htmlOutput("LRfit"),
              br(),
              
              htmlOutput("LRAUC")
              
      ),
      
      # Fifth tab content
      tabItem(tabName = "Ridge_Regreesion", h2("Ridge Regreesion"),
              
              htmlOutput("RRfit"),
              br(),
              
              htmlOutput("RRAUC"),
              br(),
              
              selectInput("RRvar", 
                          label = "Choose Lambda:",
                          choices = c("lambda.min", 
                                      "lambda.1se"),
                          selected = "lambda.min"),
              
              plotOutput("RRLambda")

      ),
      
      # Sixth tab content
      tabItem(tabName = "Lasso_Regreesion", h2("Lasso Regreesion"),
              
              htmlOutput("Lassofit"),
              br(),
              
              htmlOutput("LassoAUC"),
              br(),
              
              selectInput("Lassovar", 
                          label = "Choose Lambda:",
                          choices = c("lambda.min", 
                                      "lambda.1se"),
                          selected = "lambda.min"),
            
              plotOutput("LassoLambda")
              
      ),
      
      # Seven tab content
      tabItem(tabName = "Elastic_Net_Regreesion", h2("Elastic Net Regreesion"),
              
              htmlOutput("ENBest"),
              br(),
              
              htmlOutput("ENfit"),
              br(),
              
              htmlOutput("ENAUC"),
              br(),

              plotOutput("ENLambda")
              )
    )
  )
)

server <- function(input, output) { 
  
  library(regclass)
  library(caret)
  library(glmnet)
  library(pROC)
  library(randomForest)
  library(xgboost)
  library(Matrix)
  library(e1071)
  library(tidyverse)
  
  data(LAUNCH)
  
  LAUCH.data <- LAUNCH
  
  # Create Success
  LAUNCH$Success <- factor(ifelse(LAUNCH$Profit>4.5, "Yes","No"))
  
  # Delete Profit column
  LAUNCH$Profit <- NULL
  
  # Replace x4, x5, x7, x8, x11, x12 with their logarithms
  LAUNCH$x4 <- log10(LAUNCH$x4)
  LAUNCH$x5 <- log10(LAUNCH$x5)
  LAUNCH$x7 <- log10(LAUNCH$x7)
  LAUNCH$x8 <- log10(LAUNCH$x8)
  LAUNCH$x11 <- log10(LAUNCH$x11)
  LAUNCH$x12 <- log10(LAUNCH$x12)
  
  # for(i in c(4,5,7,8,11,12)){ LAUNCH[,i] <- log10(LAUNCH[,i])}
  
  # Remove all columns that contain a single value
  to.delete <- which(unlist(lapply(LAUNCH, function(x)length(unique(x))))==1)
  LAUNCH <- LAUNCH[,-to.delete]
  
  
  # Fit Random Forest predicting Success from all predictors on the LAUNCH dataset
  set.seed(1234)
  RF <- randomForest(Success~.,data=LAUNCH, ntree=500)
  
   # Keep the Success column and the top 100 predictors according to the random forest
  good.columns <- c("Success", names(summarize_tree(RF)$importance[1:5]))
  LAUNCH <- LAUNCH[,good.columns]
  
  # Split data into train and test
  y <- LAUNCH$Success
  LAUNCH$Success <- NULL
  
  length(summarize_tree(RF)$importance<=0.002)
  
  # Convert categorical variables into indicator variables with the folloing command
  LAUNCH <- model.matrix(~.-1,data=LAUNCH)
  
  # We want to penalty coefficient, convert the coefficient to be consistent as mean=0, sd=1, LAUNCH is matrix not data.frame
  LAUNCH <- scale(LAUNCH)
  #apply(LAUNCH,2,mean)
  #apply(LAUNCH,2,sd)
  
  set.seed(1234)
  train.rows <- sample(1:nrow(LAUNCH), floor(0.75*nrow(LAUNCH)))
  x.train <- LAUNCH[train.rows, ]
  x.test <- LAUNCH[-train.rows, ]
  
  y.train <- y[train.rows]
  y.test <- y[-train.rows]
  
  # Ridge Regression
  set.seed(1234)
  RR.cv <- cv.glmnet(x.train, y.train, alpha=0, family="binomial", nfolds=10)
  #plot(RR.cv)
  RR.cv$lambda.min #lambda with smallest estiamted error
  RR.cv$lambda.1se #lambda suggested by 1 SD rule
  
  lambda.grid <- c(0, RR.cv$lambda.min, RR.cv$lambda.1se )
  RR <- glmnet(x.train, y.train, alpha=0, lambda=lambda.grid, family="binomial", thres=1e-9) #thres only necessary if having convergence problems
  
  # Predictions for a shrinkage parameter of 0
  pred.0 <- as.numeric(predict(RR, s=0, newx=x.test, type="response")) #s: shrink parameter
  LR.AUC <- roc(y.test, pred.0)$auc
  
  # Predictions for a shrinkage parameter of lambda.min
  pred.min <- as.numeric(predict(RR, s=RR.cv$lambda.min, newx=x.test, type="response")) #s: shrink parameter
  AUC.RR.min <- roc(y.test, pred.min)$auc
  
  # Predictions for a shrinkage parameter of lambda.1se
  pred.1se <- as.numeric(predict(RR, s=RR.cv$lambda.1se, newx=x.test, type="response")) #s: shrink parameter
  AUC.RR.1se <- roc(y.test, pred.1se)$auc
  
  RR.AUC <- c(AUC.RR.1se, AUC.RR.min)
  
  # Lasso Regression
  set.seed(1234)
  Lasso.cv <- cv.glmnet(x.train, y.train, alpha=1, family="binomial", nfolds=10)
  plot(Lasso.cv)
  Lasso.cv$lambda.min #lambda with smallest estiamted error
  Lasso.cv$lambda.1se #lambda suggested by 1 SD rule
  
  lambda.grid.lasso <- c(0, Lasso.cv$lambda.min, Lasso.cv$lambda.1se )
  Lasso <- glmnet(x.train, y.train, alpha=1, lambda=lambda.grid.lasso, family="binomial", thres=1e-9) #thres only necessary if having convergence problems
  
  # Predictions for a shrinkage parameter of 0
  Lasso.pred.0 <- as.numeric(predict(Lasso, s=0, newx=x.test, type="response")) #s: shrink parameter
  roc(y.test, Lasso.pred.0)
  
  # Predictions for a shrinkage parameter of lambda.min
  Lasso.pred.min <- as.numeric(predict(Lasso, s=RR.cv$lambda.min, newx=x.test, type="response")) #s: shrink parameter
  AUC.Lasso.1se <- roc(y.test, Lasso.pred.min)$auc
  
  # Predictions for a shrinkage parameter of lambda.1se
  Lasso.pred.1se <- as.numeric(predict(Lasso, s=RR.cv$lambda.1se, newx=x.test, type="response")) #s: shrink parameter
  AUC.Lasso.min <- roc(y.test, Lasso.pred.1se)$auc
  
  Lasso.AUC <- c(AUC.Lasso.1se, AUC.Lasso.min)

  ## Elastic Net Regression
  
  # Build the model using the training set
  train.data <- cbind.data.frame(x.train, y.train)
  
  set.seed(1234)
  EN.model <- train(
    y.train ~., data =train.data, method = "glmnet",
    trControl = trainControl("cv", number = 10),
    tuneLength = 5
  )
  
  # Best tuning parameter
  ENBest <- as.numeric(EN.model$bestTune)
  
  # Coefficient of the final model. You need
  # to specify the best lambda
  ENcoef <- coef(EN.model$finalModel, EN.model$bestTune$lambda)
  
  # Model performance metrics
  predictions <-  EN.model %>% predict(x.test) %>% as.numeric()
  EN.AUC <- roc(y.test, predictions)$auc
  
  # Create reactivevalues for variables
  state <- reactiveValues()
  
  observe({
    state$xRR <- input$RRvar
    state$xLasso <- input$Lassovar
    state$yRR <- ifelse(state$xRR == 'lambda.1se', 1, 2)
    state$yLasso <- ifelse(state$xLasso == 'lambda.1se', 1, 2)
  })
  
  # output data
  output$Mydata <- renderDataTable({LAUCH.data})
  
  # output models
  
  output$LogisticsModel <- renderText(
    
    paste("<B>Logistics Regression Model:</B> ", HTML("&nbsp; SSE<sub>Logistics</sub> &nbsp;=&nbsp; &sum;&nbsp;(y-y&#770;)<sup>2</sup>"))
  )
  
  output$RidgeModel <- renderText(
    
    paste("<B>Ridge Regression Model:</B> ", HTML("&nbsp; SSE<sub>Ridge</sub> &nbsp;=&nbsp; &sum;&nbsp;(y-y&#770;)<sup>2</sup>&nbsp;+&nbsp;&lambda;&nbsp;&sum;&nbsp;&beta;<sup>2</sup>"))
  )
  
  output$LassoModel <- renderText(
    
    paste("<B>Lasso Regression Model:</B> ", HTML("&nbsp; SSE<sub>Lasso</sub> &nbsp;=&nbsp; &sum;&nbsp;(y-y&#770;)<sup>2</sup>&nbsp;+&nbsp;&lambda;&nbsp;&sum;&nbsp;|&beta;|"))
  )
  
  output$ENModel <- renderText(
    
    paste("<B>Elastic Net Regression Model:</B> ", 
          HTML("&nbsp; SSE<sub>Elastic NEt</sub> &nbsp;=&nbsp;&sum;&nbsp;(y-y&#770;)<sup>2</sup>&nbsp;+&nbsp;&lambda;&nbsp;[&nbsp;(1-&alpha;)&nbsp;&sum;&nbsp;&beta;<sup>2</sup>&nbsp;+&nbsp;&alpha;&nbsp;&sum;&nbsp;|&beta;|&nbsp;]"))
  )
  
  output$Note <- renderText(
    
    paste("<B>Note:</B> ", HTML(" &nbsp; when <b>&alpha;=0</b>, it is <b>Ridge Regression</b>; &nbsp; when <b>&alpha;=1</b>, it is <b>Lasso Regression</b>."))
  )
  
  # output Logistics Regression
  output$LRfit <- renderText(
    
    paste("<B>Logistics Model:</B> y = ", round(coef(RR)[1,3],2), "*intercept + ",
          round(coef(RR)[2,3],2), "*x4 + ",
          round(coef(RR)[3,3],2), "*x2 + ",
          round(coef(RR)[4,3],2), "*x3 + ",
          round(coef(RR)[5,3],2), "*x374 + ",
          round(coef(RR)[6,3],2), "*x5")
    
  )
  
  output$LRAUC <- renderText( paste("<B>Area under the curve:</B>", round(LR.AUC,4)))
  
  
  # output Ridge Regression
  output$RRLambda <- renderPlot(
    plot(RR.cv)
  )
  
  output$RRfit <- renderText(
    
    paste("<B>Ridge Model:</B> y = ", round(coef(RR)[1,state$yRR],2), "*intercept + ",
                               round(coef(RR)[2,state$yRR],2), "*x4 + ",
                               round(coef(RR)[3,state$yRR],2), "*x2 + ",
                               round(coef(RR)[4,state$yRR],2), "*x3 + ",
                               round(coef(RR)[5,state$yRR],2), "*x374 + ",
                               round(coef(RR)[6,state$yRR],2), "*x5")

  )
  
  output$RRAUC <- renderText( paste("<B>Area under the curve:</B>", round(RR.AUC[state$yRR],4)))
  
  
  # output Lasso Regression
  output$LassoLambda <- renderPlot(
    plot(Lasso.cv)
  )
  
  output$Lassofit <- renderText(
    
    paste("<B>Lasso Model:</B> y = ", round(coef(Lasso)[1,state$yLasso],2), "*intercept + ",
          round(coef(Lasso)[2,state$yLasso],2), "*x4 + ",
          round(coef(Lasso)[3,state$yLasso],2), "*x2 + ",
          round(coef(Lasso)[4,state$yLasso],2), "*x3 + ",
          round(coef(Lasso)[5,state$yLasso],2), "*x374 + ",
          round(coef(Lasso)[6,state$yLasso],2), "*x5")
    
  )
  
  output$LassoAUC <- renderText( paste("<B> Area under the curve: </B>", round(Lasso.AUC[state$yLasso],4)))

  # output Elastic Net Regression
  
  output$ENLambda <- renderPlot(
    plot(EN.model)
  )
  
  output$ENBest <- renderText(
    
    HTML("<B> Best alpha: </B> &nbsp;", round(ENBest[1],2), "&nbsp; &nbsp; <B>Best lambda:</B> &nbsp;", round(ENBest[2],4))
  )
  
  output$ENfit <- renderText(
    
    paste("<B>Elastic Net Model:</B> y = ", round(ENcoef[1],2), "*intercept + ",
          round(ENcoef[2],2), "*x4 + ",
          round(ENcoef[3],2), "*x2 + ",
          round(ENcoef[4],2), "*x3 + ",
          round(ENcoef[5],2), "*x374 + ",
          round(ENcoef[6],2), "*x5")
    
  )
  
  output$ENAUC <- renderText( paste("<B> Area under the curve: </B>", round(EN.AUC,4)))
  
  }
shinyApp(ui, server)
