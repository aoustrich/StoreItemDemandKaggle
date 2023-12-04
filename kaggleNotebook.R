
# Cell 1 ------------------------------------------------------------------

# library(tidyverse)
# library(tidymodels)
# library(vroom)
# library(glmnet)
# library(parallel)
# library(ranger)
# library(parallel)
# library(timetk)
# library(modeltime)
# library(forecast) 
# library(prophet) 
# library(embed)
# library(bonsai)
# library(lightgbm)
library(tidyverse)
library(tidymodels)
library(modeltime)
library(timetk)
library(vroom)
library(embed)
library(bonsai)
library(lightgbm)

train <- vroom::vroom("/kaggle/input/demand-forecasting-kernels-only/train.csv")
test <- vroom::vroom("/kaggle/input/demand-forecasting-kernels-only/test.csv")


# Cell 2 ------------------------------------------------------------------

# Set up Model, Recipe, and Workflow
boostedModel <- boost_tree(tree_depth=2, #Determined by random store-item combos
                           trees=1000,
                           learn_rate=0.01) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

itemRecipe <- recipe(sales ~ ., data = train) %>% 
  step_date(date,features = c("dow","month","year","decimal","doy", "week", "quarter")) %>%
  step_range(date_doy, min=0, max=pi) %>%
  step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy)) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(sales)) %>%
  step_rm(date, item, store) %>%
  step_normalize(all_numeric_predictors())

boost_wf <- workflow() %>%
  add_recipe(itemRecipe) %>%
  add_model(boostedModel)

# Cell 3 ------------------------------------------------------------------
nStores <- max(train$store)
nItems <- max(train$item)
for(s in 1:nStores){
  for(i in 1:nItems){
    storeItemTrain <- train %>%
      filter(store==s, item==i)
    storeItemTest <- test %>%
      filter(store==s, item==i)
    
    ## Fit the data and forecast
    fitted_wf <- boost_wf %>%
      fit(data=train)
    
    preds <- predict(fitted_wf, new_data=test) %>%
      bind_cols(test) %>%
      rename(sales=.pred) %>%
      select(id, sales)
    
    ## Save storeItem predictions
    if(s==1 & i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds, preds)
    }
    
  }
}


# Cell 4 ------------------------------------------------------------------

# Write submission
vroom_write(all_preds, file="/kaggle/working/submissions/boostedv2.csv", delim=",")
