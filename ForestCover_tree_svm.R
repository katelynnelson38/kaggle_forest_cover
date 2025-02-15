
# FOREST COVER DECISION TREE & SVM FOR SERVER

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

library(stacks)
ctrl_grid <- control_stack_grid()



# # DECISION TREE
# 
# tree_mod <- 
#   decision_tree(tree_depth = tune(),
#                 cost_complexity = tune()) %>%
#   set_engine("rpart") %>%
#   set_mode("classification")
# 
# tree_grid <- grid_regular(cost_complexity(),
#                           tree_depth(),
#                           levels = 10)
# 
# # creating a workflow to bundle the recipe and model
# tree_wf <- workflow() %>%
#   add_recipe(rec) %>%
#   add_model(tree_mod)
# 
# tree_wf
# 
# tree_res <- tree_wf %>% 
#   tune_grid(
#     resamples = folds,
#     grid = tree_grid,
#     control = ctrl_grid
#   )
# 
# tree_res %>%
#   show_best("roc_auc")
# 
# best_tree <- tree_res %>%
#   select_best("roc_auc")
# 
# final_wf <- 
#   tree_wf %>% 
#   finalize_workflow(best_tree)
# 
# final_fit <- 
#   final_wf %>%
#   last_fit(forest_split)
# 
# final_fit %>% collect_metrics()
# 
# final_tree <- extract_workflow(final_fit)
# 
# # fit the model on the entire dataset
# tree_fit <- 
#   final_tree %>%
#   fit(data = forest)
# 
# # predict on the kaggle test set
# forest_aug <- 
#   augment(tree_fit, test)
# 
# submission <- forest_aug %>% 
#   select(Id, .pred_class) %>% 
#   rename(Cover_Type = .pred_class)
# 
# # wo/ this line, some numbers are changed to scientific notation with write.csv()
# submission$Id <- as.integer(submission$Id)
# 
# write.csv(submission, 
#           "decisiontree_ForestCover.csv", 
#           row.names = FALSE)


### SVM

svm_mod <- svm_linear(cost = tune()) %>%
  set_mode('classification') %>%
  set_engine('kernlab')

svm_wf <- workflow() %>%
  add_model(svm_mod) %>%
  add_recipe(rec)

tune_grid <- grid_regular(cost(), levels = 10)

# use the same folds

library(doParallel)
cl <- makePSOCKcluster(parallel::detectCores(logical = FALSE))
registerDoParallel(cl)

start <- Sys.time()
svm_res <- tune_grid(svm_wf,
                     resamples = folds,
                     grid = tune_grid,
                     control = ctrl_grid)
Sys.time() - start

stopCluster(cl)

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
          "svm_ForestCover.csv", 
          row.names = FALSE)

# 
# save(tree_res, svm_res,
#      file = "Forest_tune.Rdata")


