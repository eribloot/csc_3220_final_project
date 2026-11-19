library(tidyverse)
library(tidymodels)
set.seed(3220)

# Load dataset, addicted_label is factor with "1" as first event level
df_labeled <- read_csv("Smartphone_Usage_And_Addiction_Analysis.csv") %>%
  select(-any_of(c("user_id", "addiction_level"))) %>%
  mutate(addicted_label = factor(addicted_label, levels = c("1", "0")))

# 80/20 Train/Test Split
data_split <- initial_split(df_labeled, prop = 0.80, strata = addicted_label)
train_data <- training(data_split)
test_data <- testing(data_split)

# Preprocessing recipe
logreg_recipe <- recipe(addicted_label ~ ., data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

# Logreg workflow
logreg_spec <- logistic_reg() %>%
  set_engine("glm") %>%
  set_mode("classification")

logreg_wf <- workflow() %>%
  add_recipe(logreg_recipe) %>%
  add_model(logreg_spec)

# Fit Model
logreg_fit <- fit(logreg_wf, data = train_data)

# Eval default (0.50 Thresh) Metrics on Test Set
baseline_metrics <- metric_set(accuracy, recall, precision, f_meas)
baseline_metrics(test_preds, truth = addicted_label, estimate = .pred_class)
roc_auc(test_preds, truth = addicted_label, .pred_1)


# Sweep prob cutoff to find threshold hitting 90-95 Recall
thresholds <- seq(0.10, 0.50, by = 0.05)

threshold_results <- map_dfr(thresholds, function(thresh) {
  test_preds %>%
    mutate(pred_custom = factor(if_else(.pred_1 >= thresh, "1", "0"), levels = c("1", "0"))) %>%
    summarise(
      threshold = thresh,
      recall    = recall_vec(addicted_label, pred_custom),
      precision = precision_vec(addicted_label, pred_custom),
      accuracy  = accuracy_vec(addicted_label, pred_custom),
      f1        = f_meas_vec(addicted_label, pred_custom)
    )
})

print(threshold_results)