library(tidyverse)
library(tidymodels)
library(vroom)
library(glmnet)
library(parallel)
library(ranger)
library(parallel)
library(timetk)
library(modeltime)
library(forecast) # for auto_arima engine
library(prophet) # for facebook prophet model engine

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

# New data with date features for 2nd Exponential Smoothing model
subTrain2 <- train %>%
  filter(store == 4, item == 15) %>% 
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




# SARIMA 1 ------------------------------------------------------------------
arima_recipe <- recipe(sales ~ ., data = subTrain) %>% 
  step_date(date,features = c("year", "doy", "week", "quarter"))

cv_split <- time_series_split(subTrain, assess="3 months", cumulative = TRUE)

S = 180

arima_model <- arima_reg(seasonal_period=S,
                         non_seasonal_ar=5, # default max p to tune
                         non_seasonal_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2 #default max D to tune
                        ) %>%
                        set_engine("auto_arima")

cl <- makePSOCKcluster(4)
doParallel::registerDoParallel(cl)

arima_wf <- workflow() %>%
  add_recipe(arima_recipe) %>%
  add_model(arima_model) %>%
  fit(data=training(cv_split))
proc.time()

## Calibrate (i.e. tune) workflow
cv_results <- modeltime_calibrate(arima_wf,
                                  new_data = testing(cv_split))

## Visualize & Evaluate CV accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## Refit best model to entire data and predict
arima_fullfit <- cv_results %>%
  modeltime_refit(data = subTrain)

p1 <- cv_results %>%
  modeltime_forecast(
    new_data = testing(cv_split),
    actual_data = subTrain) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

arima_preds <- arima_fullfit %>%
  modeltime_forecast(new_data=subTrain) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)


p2 <- arima_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = subTrain) %>%
  plot_modeltime_forecast(.interactive=FALSE)

stopCluster(cl)

# SARIMA 2 ------------------------------------------------------------------
cl <- makePSOCKcluster(4)
doParallel::registerDoParallel(cl)

arima_recipe2 <- recipe(sales ~ ., data = subTrain2) %>% 
  step_date(date,features = c("year", "doy", "week", "quarter"))

cv_split2 <- time_series_split(subTrain2, assess="3 months", cumulative = TRUE)

S = 180

arima_model2 <- arima_reg(seasonal_period=S,
                         non_seasonal_ar=5, # default max p to tune
                         non_seasonal_ma=5, # default max q to tune
                         seasonal_ar=2, # default max P to tune
                         seasonal_ma=2, #default max Q to tune
                         non_seasonal_differences=2, # default max d to tune
                         seasonal_differences=2 #default max D to tune
) %>%
  set_engine("auto_arima")

arima_wf2 <- workflow() %>%
  add_recipe(arima_recipe2) %>%
  add_model(arima_model2) %>%
  fit(data=training(cv_split2))

## Calibrate (i.e. tune) workflow
cv_results2 <- modeltime_calibrate(arima_wf2,
                                  new_data = testing(cv_split2))

## Visualize & Evaluate CV accuracy
cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

## Refit best model to entire data and predict
arima_fullfit2 <- cv_results2 %>%
  modeltime_refit(data = subTrain2)

p3 <- cv_results2 %>%
  modeltime_forecast(
    new_data = testing(cv_split2),
    actual_data = subTrain2) %>%
  plot_modeltime_forecast(.interactive=TRUE)

## Evaluate the accuracy
cv_results2 %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(.interactive = FALSE)

arima_preds2 <- arima_fullfit2 %>%
  # modeltime_forecast(h = "3 months") %>%
  modeltime_forecast(new_data = subTrain2) %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)


p4 <- arima_fullfit2 %>%
  modeltime_forecast(h = "3 months", actual_data = subTrain) %>%
  plot_modeltime_forecast(.interactive=FALSE)

stopCluster(cl)

plotly::subplot(p1, p2, p3, p4, nrows=2)


# Facebook Prophet Model 1 --------------------------------------------------

cv_split <- time_series_split(subTrain, assess="3 months", cumulative = TRUE)

prophet_model <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split))


## Cross-validate to tune model
cv_results <- modeltime_calibrate(prophet_model,
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
prophet_fullfit <- cv_results %>%
  modeltime_refit(data = subTrain)

prophet_preds <- prophet_fullfit %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

p2 <- prophet_fullfit %>%
  modeltime_forecast(h = "3 months", actual_data = subTrain) %>%
  plot_modeltime_forecast(.interactive=FALSE)

# Facebook Prophet Model 2 --------------------------------------------------

cv_split2 <- time_series_split(subTrain2, assess="3 months", cumulative = TRUE)

prophet_model2 <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(sales ~ date, data = training(cv_split2))


## Cross-validate to tune model
cv_results2 <- modeltime_calibrate(prophet_model2,
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
prophet_fullfit2 <- cv_results2 %>%
  modeltime_refit(data = subTrain2)

prophet_preds2 <- prophet_fullfit2 %>%
  modeltime_forecast(h = "3 months") %>%
  rename(date=.index, sales=.value) %>%
  select(date, sales) %>%
  full_join(., y=test, by="date") %>%
  select(id, sales)

p4 <- prophet_fullfit2 %>%
  modeltime_forecast(h = "3 months", actual_data = subTrain2) %>%
  plot_modeltime_forecast(.interactive=FALSE)


plotly::subplot(p1, p3, p2, p4, nrows=2)

