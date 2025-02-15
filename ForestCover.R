
library(tidyverse)
library(tidymodels)
library(vroom)
library(doParallel)

forest <- vroom("Forest Cover (Multinomial)/data/train.csv")
test <- vroom("Forest Cover (Multinomial)/data/test.csv")

glimpse(forest)

forest %>% group_by(Cover_Type) %>% count()

ggplot(data = forest) +
  geom_bar(aes(x = Cover_Type, fill = as.factor(Soil_Type38)))

# splitting the data into train and test sets
forest_split <- initial_split(forest)
forest_test <- testing(forest_split)
forest_train <- training(forest_split)

rec <- recipe(Cover_Type ~ ., data = forest_train) %>%
  update_role(Id, new_role = "Id") %>%
  step_mutate(Id = factor(Id)) %>%
  step_mutate_at(all_outcomes(), fn = factor, skip = TRUE) #%>%
  # step_mutate(Soil_Type2_5_6 = Soil_Type2 + Soil_Type5 + Soil_Type6,
  #             Soil_Type10_11 = Soil_Type10 + Soil_Type11,
  #             Soil_Type29_30 = Soil_Type29 + Soil_Type30,
  #             Soil_Type39_40 = Soil_Type39 + Soil_Type40,
  #             Soil_Type22t28_31_33_38 = Soil_Type22 + Soil_Type23 +
  #               Soil_Type24 + Soil_Type25 + Soil_Type26 + Soil_Type27 +
  #               Soil_Type28 + Soil_Type31 + Soil_Type33 + Soil_Type38) %>%
  # step_rm(Soil_Type2, Soil_Type5, Soil_Type6,
  #         Soil_Type10, Soil_Type11,
  #         Soil_Type29, Soil_Type30,
  #         Soil_Type39, Soil_Type40,
  #         Soil_Type22, Soil_Type23, Soil_Type24, Soil_Type25,
  #         Soil_Type26, Soil_Type27, Soil_Type28, Soil_Type31,
  #         Soil_Type33, Soil_Type38) %>%
  #step_nzv(all_numeric_predictors(), freq_cut = 99.5/.5) %>%
  #step_corr() #%>% # I don't think this removes anything
  # step_mutate_at(contains("Wilderness"), fn = factor) %>%
  # step_mutate_at(contains("Soil_Type"), fn = factor) %>%
  #step_normalize(all_numeric_predictors()) %>%
  #step_pca(all_numeric_predictors(), threshold = .8)

prep <- prep(rec)
baked <- bake(prep, new_data = NULL)
dim(baked)
glimpse(baked)

folds <- vfold_cv(forest_train, v = 5)

library(stacks)
ctrl_grid <- control_stack_grid()

apply(forest, MARGIN = 2, FUN = sum)

## LOGISTIC REGRESSION

lr_mod <- multinom_reg(penalty = tune(), mixture = tune()) %>%
  set_mode('classification') %>%
  set_engine('glmnet')

lr_wf <- workflow() %>%
  add_model(lr_mod) %>%
  add_recipe(rec)

lr_wf

tune_grid <- grid_regular(penalty(), mixture(), levels = 10)

# use the same folds

# cl <- makePSOCKcluster(parallel::detectCores(logical = FALSE))
# registerDoParallel(cl)

start <- Sys.time()
lr_res <- tune_grid(lr_wf,
                    resamples = folds,
                    grid = tune_grid,
                    control = ctrl_grid)
Sys.time() - start

# stopCluster(cl)

lr_res %>% show_best('accuracy')

best_lr <- lr_res %>% select_best('accuracy')

final_wf <- 
  lr_wf %>%
  finalize_workflow(best_lr)

final_fit <- 
  final_wf %>%
  last_fit(forest_split)

final_fit %>% collect_metrics()

# extract the final model (must do this to predict on new data)
final_lr <- extract_workflow(final_fit)

# fit the tuned model on the entire dataset
lr_fit <- 
  final_lr %>%
  fit(data = forest)

# predict on the kaggle test set
forest_aug <- 
  augment(lr_fit, test)

submission <- forest_aug %>% 
  select(Id, .pred_class) %>% 
  rename(Cover_Type = .pred_class)

# wo/ this line, some numbers are changed to scientific notation with write.csv()
submission$Id <- as.integer(submission$Id)

write.csv(submission, 
          "Forest Cover (Multinomial)/lr_ForestCover.csv", 
          row.names = FALSE)


## NAIVE BAYES

nb_rec <- rec %>%
  step_normalize()

library(discrim) # required to use the naivebayes engine

nb_mod <- naive_Bayes(smoothness = tune(), Laplace = tune()) %>%
  set_mode('classification') %>%
  set_engine('naivebayes')

nb_wf <- workflow() %>%
  add_model(nb_mod) %>%
  add_recipe(nb_rec)

nb_wf

tune_grid <- grid_regular(smoothness(), Laplace(), levels = 10)

# use the same folds

# cl <- makePSOCKcluster(parallel::detectCores(logical = FALSE))
# registerDoParallel(cl)

start <- Sys.time()
nb_res <- tune_grid(nb_wf,
                    resamples = folds,
                    grid = tune_grid,
                    control = ctrl_grid)
Sys.time() - start

# stopCluster(cl)

nb_res %>% show_best('accuracy')

best_nb <- nb_res %>% select_best('accuracy')

final_wf <- 
  nb_wf %>%
  finalize_workflow(best_nb)

final_fit <- 
  final_wf %>%
  last_fit(forest_split)

final_fit %>% collect_metrics()

# extract the final model (must do this to predict on new data)
final_nb <- extract_workflow(final_fit)

# fit the tuned model on the entire dataset
nb_fit <- 
  final_nb %>%
  fit(data = forest)

# predict on the kaggle test set
forest_aug <- 
  augment(nb_fit, test)

submission <- forest_aug %>% 
  select(Id, .pred_class) %>% 
  rename(Cover_Type = .pred_class)

# wo/ this line, some numbers are changed to scientific notation with write.csv()
submission$Id <- as.integer(submission$Id)

write.csv(submission, 
          "Forest Cover (Multinomial)/nb_ForestCover.csv", 
          row.names = FALSE)


# DECISION TREE

tree_mod <- 
  decision_tree(tree_depth = tune(),
                cost_complexity = tune()) %>%
  set_engine("rpart") %>%
  set_mode("classification")

tree_grid <- grid_regular(cost_complexity(),
                          tree_depth(),
                          levels = 10)

# creating a workflow to bundle the recipe and model
tree_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(tree_mod)

tree_wf

tree_res <- tree_wf %>% 
  tune_grid(
    resamples = folds,
    grid = tree_grid,
    control = ctrl_grid
  )

tree_res %>%
  show_best("roc_auc")

best_tree <- tree_res %>%
  select_best("roc_auc")

final_wf <- 
  tree_wf %>% 
  finalize_workflow(best_tree)

final_fit <- 
  final_wf %>%
  last_fit(forest_split)

final_fit %>% collect_metrics()

final_tree <- extract_workflow(final_fit)

# fit the model on the entire dataset
tree_fit <- 
  final_tree %>%
  fit(data = forest)

# predict on the kaggle test set
forest_aug <- 
  augment(tree_fit, test)

submission <- forest_aug %>% 
  select(Id, .pred_class) %>% 
  rename(Cover_Type = .pred_class)

# wo/ this line, some numbers are changed to scientific notation with write.csv()
submission$Id <- as.integer(submission$Id)

write.csv(submission, 
          "decisiontree_ForestCover.csv", 
          row.names = FALSE)


### RF

rf_mod <- rand_forest(mtry = tune(), min_n = tune()) %>%
  set_engine('ranger') %>%
  set_mode('classification')

rf_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(rf_mod)

rf_wf

# cl <- makePSOCKcluster(parallel::detectCores(logical = FALSE))
# registerDoParallel(cl)

start <- Sys.time()
rf_res <- tune_grid(rf_wf,
                    resamples = folds,
                    grid = 5, # grid_regular doesn't work with rf mtry
                    control = ctrl_grid)
Sys.time() - start

# stopCluster(cl)

rf_res %>% show_best('roc_auc')

best_rf <- rf_res %>% select_best('roc_auc')

final_wf <- rf_wf %>%
  finalize_workflow(best_rf)

final_fit <- final_wf %>%
  last_fit(forest_split)

final_fit %>% collect_metrics()

final_rf <- extract_workflow(final_fit)

# fit the model with the entire dataset
rf_fit <- 
  final_rf %>%
  fit(data = forest)

# predict on the kaggle test set
forest_aug <- 
  augment(rf_fit, test)

submission <- forest_aug %>% 
  select(Id, .pred_class) %>% 
  rename(Cover_Type = .pred_class)

# wo/ this line, some numbers are changed to scientific notation with write.csv()
submission$Id <- as.integer(submission$Id)

write.csv(submission, 
          "Forest Cover (Multinomial)/rf_ForestCover.csv", 
          row.names = FALSE)


## XGBoost

xgb_mod <- boost_tree(learn_rate = tune(),
                      min_n = tune(),
                      mtry = tune(),
                      tree_depth = tune(),
                      sample_size = tune()) %>%
  set_engine('xgboost') %>%
  set_mode('classification')

xgb_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(xgb_mod)

# cl <- makePSOCKcluster(parallel::detectCores(logical = FALSE))
# registerDoParallel(cl)

start <- Sys.time()
xgb_res <- tune_grid(xgb_wf,
                    resamples = folds,
                    grid = 7, # grid_regular doesn't work with rf mtry
                    control = ctrl_grid)
Sys.time() - start

# stopCluster(cl)

xgb_res %>% show_best('roc_auc')

best_xgb <- xgb_res %>% select_best('roc_auc')

final_wf <- xgb_wf %>%
  finalize_workflow(best_xgb)

final_fit <- final_wf %>%
  last_fit(forest_split)

final_fit %>% collect_metrics()

final_xgb <- extract_workflow(final_fit)

# fit the model with the entire dataset
xgb_fit <- 
  final_xgb %>%
  fit(data = forest)

# predict on the kaggle test set
forest_aug <- 
  augment(xgb_fit, test)

submission <- forest_aug %>% 
  select(Id, .pred_class) %>% 
  rename(Cover_Type = .pred_class)

# wo/ this line, some numbers are changed to scientific notation with write.csv()
submission$Id <- as.integer(submission$Id)

write.csv(submission, 
          "Forest Cover (Multinomial)/xgb_ForestCover.csv", 
          row.names = FALSE)


## KNN

knn_mod <- nearest_neighbor(neighbors = tune()) %>%
  set_mode('classification') %>%
  set_engine('kknn')

knn_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(knn_mod)

knn_wf

tune_grid <- grid_regular(neighbors(), levels = 10)

# cl <- makePSOCKcluster(parallel::detectCores(logical = FALSE))
# registerDoParallel(cl)

start <- Sys.time()
knn_res <- tune_grid(knn_wf,
                     resamples = folds,
                     grid = tune_grid,
                     control = ctrl_grid)
Sys.time() - start

# stopCluster(cl)

knn_res %>% show_best('roc_auc')

best_knn <- knn_res %>% select_best('roc_auc')

final_wf <- knn_wf %>%
  finalize_workflow(best_knn)

final_fit <- final_wf %>%
  last_fit(forest_split)

final_fit %>% collect_metrics()

final_knn <- extract_workflow(final_fit)

# fit the model on the entire dataset
knn_fit <- 
  final_knn %>%
  fit(data = forest)

# predict on the kaggle test set
forest_aug <- 
  augment(knn_fit, test)

submission <- forest_aug %>% 
  select(Id, .pred_class) %>% 
  rename(Cover_Type = .pred_class)

# wo/ this line, some numbers are changed to scientific notation with write.csv()
submission$Id <- as.integer(submission$Id)

write.csv(submission, 
          "Forest Cover (Multinomial)/knn_ForestCover.csv", 
          row.names = FALSE)


### NNET

nnet_mod <- mlp(penalty = tune()) %>%
  set_mode('classification') %>%
  set_engine('keras')

nnet_wf <- workflow() %>%
  add_model(nnet_mod) %>%
  add_recipe(rec)

tune_grid <- grid_regular(penalty(), levels = 10)

# use the same folds

start <- Sys.time()
nnet_res <- tune_grid(nnet_wf,
                      resamples = folds,
                      grid = tune_grid)
Sys.time() - start

nnet_res %>% show_best('roc_auc')

best_nnet <- nnet_res %>% select_best('roc_auc')

final_wf <- 
  nnet_wf %>%
  finalize_workflow(best_nnet)

final_fit <- 
  final_wf %>%
  last_fit(forest_split)

final_fit %>% collect_metrics()

final_nnet <- extract_workflow(final_fit)

# fit the model on the entire dataset
nnet_fit <- 
  final_nnet %>%
  fit(data = forest)

# predict on the kaggle test set
forest_aug <- 
  augment(nnet_fit, test)

submission <- forest_aug %>% 
  select(Id, .pred_class) %>% 
  rename(Cover_Type = .pred_class)

# wo/ this line, some numbers are changed to scientific notation with write.csv()
submission$Id <- as.integer(submission$Id)

write.csv(submission, 
          "Forest Cover (Multinomial)/nnet_ForestCover.csv", 
          row.names = FALSE)


### SVM

svm_mod <- svm_linear(cost = tune()) %>%
  set_mode('classification') %>%
  set_engine('kernlab')

svm_wf <- workflow() %>%
  add_model(svm_mod) %>%
  add_recipe(rec)

tune_grid <- grid_regular(cost(), levels = 10)

# use the same folds

# library(doParallel)
# cl <- makePSOCKcluster(parallel::detectCores(logical = FALSE))
# registerDoParallel(cl)

start <- Sys.time()
svm_res <- tune_grid(svm_wf,
                     resamples = folds,
                     grid = tune_grid)
Sys.time() - start

# stopCluster(cl)

svm_res %>% show_best('roc_auc')

best_svm <- svm_res %>% select_best('roc_auc')

final_wf <- 
  svm_wf %>%
  finalize_workflow(best_svm)

final_fit <- 
  final_wf %>%
  last_fit(forest_split)

final_fit %>% collect_metrics()

final_svm <- extract_workflow(final_fit)

# fit the model on the entire dataset
svm_fit <- 
  final_svm %>%
  fit(data = forest)

# predict on the kaggle test set
forest_aug <- 
  augment(svm_fit, test)

submission <- forest_aug %>% 
  select(Id, .pred_class) %>% 
  rename(Cover_Type = .pred_class)

# wo/ this line, some numbers are changed to scientific notation with write.csv()
submission$Id <- as.integer(submission$Id)

write.csv(submission, 
          "Forest Cover (Multinomial)/svm_ForestCover.csv", 
          row.names = FALSE)



#######################
### STACKED MODEL

library(stacks)

forest_stack <- stacks() %>%
  #add_candidates(lr_res) %>%
  add_candidates(nb_res) %>%
  #add_candidates(knn_res) %>%
  add_candidates(rf_res) %>%
  add_candidates(xgb_res)

stack_mod <-
  forest_stack %>%
  blend_predictions() %>%
  fit_members()

autoplot(stack_mod)
autoplot(stack_mod, type = 'weights')

forest_aug <- 
  augment(stack_mod, test)

preds_only <- forest_aug %>% select(Id, .pred_class) %>% rename(Cover_Type = .pred_class)

# wo/ this line, some numbers are changed to scientific notation with write.csv()
preds_only$Id <- as.integer(preds_only$Id)

write.csv(preds_only, 
          "Forest Cover (Multinomial)/stacked_ForestCover.csv", 
          row.names = FALSE)
