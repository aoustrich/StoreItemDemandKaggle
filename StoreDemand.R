library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(parallel)
library(ranger)
library(parallel)
library(timetk)
library(modeltime)

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

# New data with date features
subTrain2 <- train %>%
  filter(store == 7, item == 17) %>% 
  select(-store, -item)

subRecipe2 <- recipe(sales ~ ., data = subTrain2) %>% 
  step_date(date,features = c("year", "doy", "week", "quarter"))

prepped2 <- prep(subRecipe2)
baked2 <- bake(prepped2, new_data = subTrain2)

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
  

# Exponential Smoothing 1 -------------------------------------------------
# Create cross validation split 
cv_split <- time_series_split(subTrain, assess="3 months", cumulative = TRUE)

es_model <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data=training(cv_split))

## Cross-validate to tune model
cv_results <- modeltime_calibrate(es_model,
                                 new_data = testing(cv_split))
## Visualize CV results (top row)
p1 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = subTrain) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## Refit to all data then forecast
es_fullfit <- cv_results %>%
  modeltime_refit(data = subTrain)

es_preds <- es_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

p2 <- es_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = subTrain) %>%
  plot_modeltime_forecast(.interactive=FALSE)






  






# Exponential Smoothing 2 -------------------------------------------------
cv_split2 <- time_series_split(subTrain2, assess="3 months", cumulative = TRUE)

es_model2 <- exp_smoothing() %>%
  set_engine("ets") %>%
  fit(sales~date, data=training(cv_split2))

## Cross-validate to tune model
cv_results2 <- modeltime_calibrate(es_model2,
                                  new_data = testing(cv_split2))
## Visualize CV results (top row)
p3 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = subTrain2) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## Refit to all data then forecast
es_fullfit2 <- cv_results2 %>%
  modeltime_refit(data = subTrain2)

es_preds2 <- es_fullfit2 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

p4 <- es_fullfit2 %>%
  modeltime_forecast(h = "3 months", actual_data = subTrain2) %>%
  plot_modeltime_forecast(.interactive=FALSE)

plotly::subplot(p1, p2, p3, p4, nrows=2)
