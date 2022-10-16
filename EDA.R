library(pacman)

p_load(data.table)
p_load(ggplot2)
p_load(tidymodels)



raw_dataframe = fread('C:\\Users\\pedro_jw08iyg\\OneDrive\\Área de Trabalho\\DSA\\Projetos\\BigDataRealTimeAnalyticscomPythoneSpark\\Projeto4\\projeto4_telecom_treino.csv')
raw_dataframe[,churn:=factor(churn, levels = c("yes",'no'))]

predictive_variables = c('international_plan','voice_mail_plan','number_vmail_messages',
  'total_day_minutes','total_day_charge','total_eve_minutes','total_eve_charge',
  'total_night_minutes','total_night_charge','total_intl_minutes','total_intl_calls',
  'total_intl_charge','number_customer_service_calls')


cleaned_dataframe_training = copy(raw_dataframe)
cleaned_dataframe_training = raw_dataframe[,.SD,.SDcols = c('churn',predictive_variables)]


# 1) EDA AND FEATURE ENGINEERING ----

# 1.1) number_vmail_messages

ggplot(raw_dataframe) +
  aes(x = number_vmail_messages, fill = churn) +
  geom_histogram(bins = 30L) +
  scale_fill_hue(direction = 1) +
  theme_minimal()

ggplot(raw_dataframe[number_vmail_messages>0]) +
  aes(x = number_vmail_messages, fill = churn, alpha = 0.5) +
  geom_density() +
  scale_fill_hue(direction = 1) +
  theme_minimal()

#there's a high concentration of the variable on 0. 
#the distribution on the non-zero range has no clear difference for churning status 



# 1.2) number_customer_service_calls
ggplot(raw_dataframe) +
  aes(x = factor(number_customer_service_calls), fill = churn) +
  geom_bar() +
  scale_fill_hue(direction = 1) +
  theme_minimal()


ggplot(raw_dataframe) +
  aes(x = factor(number_customer_service_calls), fill = churn) +
  geom_bar(position = "fill") +
  theme_minimal()

#there´s a significant increase on the churning rate for number of calls more than 4



#1.2) total_intl_minutes
ggplot(raw_dataframe) +
  aes(x = total_intl_minutes, fill = churn) +
  geom_histogram(bins = 30L) +
  scale_fill_hue(direction = 1) +
  theme_minimal() +
  facet_grid(rows =vars(churn))

ggplot(raw_dataframe) +
  aes(x = total_intl_minutes, fill = churn, alpha = 0.5) +
  geom_density()+
  theme_minimal()
 #No meaningful differences for churning status regarding distribution
 #only the non-churn has zero values, which are in high concentration
   
  
  #1.3) total_intl_calls
  ggplot(raw_dataframe) +
    aes(x = total_intl_calls, fill = churn) +
    geom_histogram(bins = 30L) +
    scale_fill_hue(direction = 1) +
    theme_minimal()
  
  ggplot(raw_dataframe) +
    aes(x = total_intl_calls, fill = churn, alpha = 0.5) +
    geom_density()+
  theme_minimal()
  
  ggplot(raw_dataframe) +
    aes(y = total_intl_calls, x = churn ,fill = churn) +
    geom_boxplot()
  
  #non-churn has a slighly higher difference 
  
  
#1.4) Transforming number_vmail_messages variable AUC = 0.805
cleaned_dataframe_training[,number_vmail_messages:=ifelse(number_vmail_messages==0,'equal_zero','more_than_zero')]
cleaned_dataframe_training[,number_vmail_messages:=as.factor(number_vmail_messages)]


#1.5) Transfoming number_customer_service_calls AUC = 0.839
cleaned_dataframe_training[,number_customer_service_calls:=ifelse(number_customer_service_calls<4,'less_4','equal_more_than_4')]
cleaned_dataframe_training[,number_customer_service_calls:=as.factor(number_customer_service_calls)]

#1.6) Excluding total_intl_minutes #auc = 0.839
cleaned_dataframe_training[,total_intl_minutes:=NULL]

#1.7) Agreggating similar variables AUC = 0.84
cleaned_dataframe_training[,total_minutes:=total_day_minutes+total_eve_minutes+total_night_minutes]
cleaned_dataframe_training[,total_day_minutes:=NULL]
cleaned_dataframe_training[,total_eve_minutes:=NULL]
cleaned_dataframe_training[,total_night_minutes:=NULL]

cleaned_dataframe_training[,total_charge:=total_day_charge+total_eve_charge+total_night_charge]
cleaned_dataframe_training[,total_day_charge:=NULL]
cleaned_dataframe_training[,total_eve_charge:=NULL]
cleaned_dataframe_training[,total_night_charge:=NULL]


#1.8) High collinearity with total_charge and total_minutes
ggplot(cleaned_dataframe_training) +
  aes(x = total_charge, y = total_minutes, colour = churn) +
  geom_point(shape = "circle", size = 1.5, alpha = 0.5) +
  scale_color_hue(direction = 1) +
  theme_minimal()


#1.9) Removing Total Charge AUC = 0.84
cleaned_dataframe_training[,total_charge:=NULL]


#1.10) Excluding number_vmail_messages AUC = 0.84
cleaned_dataframe_training[,number_vmail_messages:=NULL]


#1.12) total_intl_calls

ggplot(cleaned_dataframe_training) +
aes(x = total_intl_calls, fill = churn, alpha = 0.5) +
geom_density()+
theme_minimal()


ggplot(cleaned_dataframe_training) +
  aes(y = total_intl_calls,x = churn, fill = churn, alpha = 0.5) +
  geom_boxplot()+
theme_minimal()
#total_intl_calls has low influence on churn


#Excluding total_intl_calls AUC = 0.838
cleaned_dataframe_training[,total_intl_calls:=NULL]


# number_customer_service_calls x voice_mail_plan
ggplot(cleaned_dataframe_training) +
  aes(
    x = number_customer_service_calls,
    fill = voice_mail_plan
  ) +
  geom_bar(position = 'fill') +
  scale_fill_hue(direction = 1) +
  theme_minimal()


prop.table(table(cleaned_dataframe_training[,.(number_customer_service_calls,voice_mail_plan)]))


#total_intl_charge
ggplot(cleaned_dataframe_training) +
  aes(x = total_intl_charge, fill = churn, alpha = 0.5) +
  geom_density() +
  scale_fill_hue(direction = 1) +
  theme_minimal()

#low association with target variable AUC = 0.835
cleaned_dataframe_training[,total_intl_charge:=NULL]








# 2) DATA RESAMPLING ----


raw_dataframe_test = fread('C:\\Users\\pedro_jw08iyg\\OneDrive\\Área de Trabalho\\DSA\\Projetos\\BigDataRealTimeAnalyticscomPythoneSpark\\Projeto4\\projeto4_telecom_teste.csv')

raw_dataframe_test = raw_dataframe_test[,.SD,.SDcols = c('churn',predictive_variables)]

steps_manual_cleaning = function (DT) {
  

  DT[,churn:=factor(churn, levels = c("yes",'no'))]
  #DT = DT[,.SD,.SDcols = c('churn',predictive_variables)]
  #filtro_colunas = c('churn',predictive_variables)
  #print(filtro_colunas)
  #DT = DT[,..filtro_colunas]
  #DT[, filtro_colunas, with=FALSE]
  DT[,number_vmail_messages:=ifelse(number_vmail_messages==0,'equal_zero','more_than_zero')]
  DT[,number_vmail_messages:=as.factor(number_vmail_messages)]
  DT[,number_customer_service_calls:=ifelse(number_customer_service_calls<4,'less_4','equal_more_than_4')]
  DT[,number_customer_service_calls:=as.factor(number_customer_service_calls)]
  DT[,total_intl_minutes:=NULL]
  DT[,total_minutes:=total_day_minutes+total_eve_minutes+total_night_minutes]
  DT[,total_day_minutes:=NULL]
  DT[,total_eve_minutes:=NULL]
  DT[,total_night_minutes:=NULL]
  DT[,total_charge:=total_day_charge+total_eve_charge+total_night_charge]
  DT[,total_day_charge:=NULL]
  DT[,total_eve_charge:=NULL]
  DT[,total_night_charge:=NULL]
  DT[,total_charge:=NULL]
  DT[,number_vmail_messages:=NULL]
  DT[,total_intl_calls:=NULL]
  DT[,total_intl_charge:=NULL] 
  
  #print(head(DT))
  
  return(DT)

  }

cleaned_dataframe_testing = steps_manual_cleaning(raw_dataframe_test)

#3) MODEL SPECIFICATION ----

baseline_model = logistic_reg() |> 
  # Set the engine
  set_engine("glm") |> 
  # Set the mode
  set_mode("classification")



#4) FEATURE ENGINEERING ----

recipe_baseline = recipes::recipe(churn~.,
                                  data = cleaned_dataframe_training) |>
  step_normalize(all_numeric(),-all_outcomes()) |>
  step_dummy(all_nominal(),-all_outcomes())


#5) RECIPE TRAINING ----

recipe_prep_baseline = recipe_baseline |> 
  prep(training = cleaned_dataframe_training)

#6) PREPROCESS TRAINING DATA ----

cleaned_dataframe_training_prep = recipe_prep_baseline |>
  recipes::bake(new_data = NULL)


#7) PREPROCESS TEST DATA ----

cleaned_dataframe_testing_prep = recipe_prep_baseline |> 
  recipes::bake(new_data = cleaned_dataframe_testing)


#8) MODELS FITTING ----

baseline_model_fit = baseline_model |>
  parsnip::fit(churn ~ .,
               data = cleaned_dataframe_training_prep)




parsnip::tidy(baseline_model_fit)
parsnip::glance(baseline_model_fit)

#8) PREDICTIONS ON TEST DATA ----

predictions_baseline = predict(baseline_model_fit,
                                      new_data = cleaned_dataframe_testing_prep)


setDT(predictions_baseline)

predictions_baseline = data.table(predictions_baseline,true_class = cleaned_dataframe_testing_prep$churn)

#confusion matrix
conf_mat(data = predictions_baseline,
         estimate = .pred_class,
         truth = true_class)

#probability
proba_temp = predict(baseline_model_fit,
                     new_data = cleaned_dataframe_testing_prep,
                     type = 'prob')



predicion_proba_positive_class = as.data.table(proba_temp[,1])
setnames(predicion_proba_positive_class, new = 'predicted_proba')

predictions_baseline = data.table(predictions_baseline, predicion_proba_positive_class)
rm(proba_temp)

#AUC 
yardstick::roc_auc(data = predictions_baseline,
                  estimate = predicted_proba,
                  truth = true_class)

autoplot(yardstick::roc_curve(data = predictions_baseline,
                              estimate = predicted_proba,
                              truth = true_class))

