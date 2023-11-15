library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(parallel)
library(ranger)
library(parallel)

# Load data
train <- vroom("data/train.csv")
test <- vroom("data/test.csv")

# Choose one store/item combination for ease of testing (SUBset of TRAINing data = subTrain)
subTrain <- train %>%
  filter(store == 3, item == 14) %>% 
  select(-store, -item)

# Recipe for time series data
subRecipe <- recipe(sales ~ ., data = subTrain) %>% 
  step_date(date,features = c("year", "doy", "week", "quarter")) #%>%
  # step_range(date_doy, min=0, max=pi) %>%
  # step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy))


prepped <- prep(subRecipe)
baked <- bake(prepped, new_data = subTrain)

# # Create time series cross validation folds
# subFolds <- sliding_period(subTrain, date, period = "1 year", 
#                            cumulative = TRUE, 
#                            skip = "1 day")


# Random Forest -----------------------------------------------------------
randForestModel <- rand_forest(mtry = tune(),
                               min_n=tune(),
                               trees=500) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

forestWF <- workflow() %>% 
  add_recipe(subRecipe) %>% 
  add_model(randForestModel)

# create tuning grid
forest_tuning_grid <- grid_regular(mtry(range = c(1,(ncol(baked)-1))),
                                   min_n(),
                                   levels = 10)

# split data for cross validation
rfolds <- vfold_cv(subTrain, v = 5, repeats=1)


cl <- makePSOCKcluster(4)
doParallel::registerDoParallel(cl)
# run cross validation
treeCVResults <- forestWF %>% 
  tune_grid(resamples = rfolds,
            grid = forest_tuning_grid,
            metrics=metric_set(smape)) 
proc.time()

stopCluster(cl)

# select best model
best_tuneForest <- treeCVResults %>% 
  select_best("smape")

collect_metrics(treeCVResults) %>% 
  filter(mtry == best_tuneForest$mtry,
         min_n == best_tuneForest$min_n) %>%
  pull("mean")
  





