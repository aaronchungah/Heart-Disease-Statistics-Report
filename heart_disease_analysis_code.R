set.seed(1101)

library(class)
library(rpart)
library(rpart.plot)
library(e1071)
library(ROCR)
library(arules)
library(arulesViz)

# =========================
# LOAD DATA
# =========================
df <- read.csv("heart-disease-dsa1101.csv")

# =========================
# DATA CLEANING
# =========================
mode_val <- names(sort(table(df$blood.disorder[df$blood.disorder != 0]), decreasing = TRUE))[1]
df$blood.disorder[df$blood.disorder == 0] <- as.numeric(mode_val)

df$sex <- factor(df$sex)
df$chest.pain <- factor(df$chest.pain)
df$fbs <- factor(df$fbs)
df$rest.ecg <- factor(df$rest.ecg)
df$angina <- factor(df$angina)
df$blood.disorder <- factor(df$blood.disorder)
df$disease <- factor(df$disease)

# numeric outcome for ROC/AUC where needed
y_num <- as.numeric(as.character(df$disease))

# =========================
# EDA
# Topic 2: summaries, histograms, boxplots, scatterplots
# =========================
summary(df)

# response distribution
barplot(table(df$disease),
        col = "skyblue",
        main = "Heart Disease Distribution",
        ylab = "Count")

# histograms
numeric_vars <- c("age","bp","chol","heart.rate","st.depression","vessels")

par(mfrow = c(2, 3))
for (v in numeric_vars) {
  hist(df[[v]],
       main = paste("Histogram of", v),
       xlab = v,
       col = "lightblue")
}
par(mfrow = c(1, 1))

# boxplots by disease
par(mfrow = c(2, 3))
for (v in numeric_vars) {
  boxplot(df[[v]] ~ df$disease,
          main = paste(v, "by Disease"),
          xlab = "Disease",
          ylab = v,
          col = c("orange", "green"))
}
par(mfrow = c(1, 1))

# scatterplots for quantitative association
par(mfrow = c(2, 3))
plot(df$age, df$heart.rate, pch = 20, col = "darkblue",
     main = "Age vs Heart Rate", xlab = "age", ylab = "heart.rate")
plot(df$age, df$st.depression, pch = 20, col = "darkblue",
     main = "Age vs ST Depression", xlab = "age", ylab = "st.depression")
plot(df$bp, df$chol, pch = 20, col = "darkblue",
     main = "BP vs Chol", xlab = "bp", ylab = "chol")
plot(df$bp, df$heart.rate, pch = 20, col = "darkblue",
     main = "BP vs Heart Rate", xlab = "bp", ylab = "heart.rate")
plot(df$chol, df$heart.rate, pch = 20, col = "darkblue",
     main = "Chol vs Heart Rate", xlab = "chol", ylab = "heart.rate")
plot(df$st.depression, df$heart.rate, pch = 20, col = "darkblue",
     main = "ST Depression vs Heart Rate", xlab = "st.depression", ylab = "heart.rate")
par(mfrow = c(1, 1))

# categorical vs disease
categorical_vars <- c("sex","chest.pain","fbs","rest.ecg","angina","blood.disorder")

par(mfrow = c(2, 3))
for (v in categorical_vars) {
  tab <- prop.table(table(df[[v]], df$disease), 1)
  barplot(t(tab),
          beside = FALSE,
          col = c("lightblue", "pink"),
          main = paste("Disease Proportion by", v),
          ylab = "Proportion")
}
par(mfrow = c(1, 1))

# =========================
# PREPARE DATA FOR KNN
# Topic 4: knn() requires matrices of predictors and class labels
# =========================
x <- model.matrix(disease ~ . - 1, data = df)
x_scaled <- scale(x)
y <- df$disease

# =========================
# 5-FOLD CV
# Topic 4: N-fold cross-validation
# =========================
n <- nrow(df)
folds <- split(sample(1:n), rep(1:5, length.out = n))

# TPR and Precision from confusion matrix
get_tpr <- function(actual, pred) {
  tab <- table(actual, pred)
  if (nrow(tab) < 2 || ncol(tab) < 2) return(NA)
  TP <- tab["1", "1"]
  FN <- tab["1", "0"]
  TP / (TP + FN)
}

get_precision <- function(actual, pred) {
  tab <- table(actual, pred)
  if (nrow(tab) < 2 || ncol(tab) < 2) return(NA)
  TP <- tab["1", "1"]
  FP <- tab["0", "1"]
  TP / (TP + FP)
}

# =========================
# KNN TUNING
# =========================
k_vals <- seq(3, 21, 2)
knn_tpr <- numeric(length(k_vals))

for (i in 1:length(k_vals)) {
  tprs <- c()
  
  for (f in folds) {
    test <- f
    train <- setdiff(1:n, test)
    
    pred <- knn(train = x_scaled[train, ],
                test = x_scaled[test, ],
                cl = y[train],
                k = k_vals[i])
    
    tprs <- c(tprs, get_tpr(y[test], pred))
  }
  
  knn_tpr[i] <- mean(tprs, na.rm = TRUE)
}

best_k <- k_vals[which.max(knn_tpr)]

plot(k_vals, knn_tpr, type = "b",
     main = "KNN Tuning",
     xlab = "k",
     ylab = "Average TPR")

# =========================
# DECISION TREE TUNING
# Topic 5 uses rpart + minsplit + information split
# =========================
ms_vals <- c(5, 10, 15, 20, 25)
dt_tpr <- numeric(length(ms_vals))

for (i in 1:length(ms_vals)) {
  tprs <- c()
  
  for (f in folds) {
    test <- f
    train <- setdiff(1:n, test)
    
    fit <- rpart(disease ~ .,
                 method = "class",
                 data = df[train, ],
                 control = rpart.control(minsplit = ms_vals[i]),
                 parms = list(split = "information"))
    
    pred <- predict(fit, df[test, ], type = "class")
    tprs <- c(tprs, get_tpr(y[test], pred))
  }
  
  dt_tpr[i] <- mean(tprs, na.rm = TRUE)
}

best_ms <- ms_vals[which.max(dt_tpr)]

plot(ms_vals, dt_tpr, type = "b",
     main = "Decision Tree Tuning",
     xlab = "minsplit",
     ylab = "Average TPR")

# =========================
# FINAL MODELS
# =========================

# KNN
knn_pred <- knn(train = x_scaled,
                test = x_scaled,
                cl = y,
                k = best_k,
                prob = TRUE)

knn_raw_prob <- attr(knn_pred, "prob")
knn_prob <- ifelse(knn_pred == "1", knn_raw_prob, 1 - knn_raw_prob)

# Decision Tree
dt_final <- rpart(disease ~ .,
                  method = "class",
                  data = df,
                  control = rpart.control(minsplit = best_ms),
                  parms = list(split = "information"))

dt_pred <- predict(dt_final, df, type = "class")
dt_prob <- predict(dt_final, df, type = "prob")[, "1"]

rpart.plot(dt_final, type = 4, extra = 2, clip.right.labs = FALSE)

# Logistic Regression
lr_final <- glm(disease ~ ., data = df, family = binomial(link = "logit"))
lr_prob <- predict(lr_final, type = "response")
lr_pred <- ifelse(lr_prob > 0.5, 1, 0)
lr_pred <- factor(lr_pred)

# Naive Bayes
nb_final <- naiveBayes(disease ~ ., data = df)
nb_pred <- predict(nb_final, df)
nb_raw <- predict(nb_final, df, type = "raw")
nb_prob <- nb_raw[, "1"]

# =========================
# CONFUSION MATRICES, TPR, PRECISION
# Topic 4 diagnostics
# =========================
cat("Confusion Matrix: KNN\n")
print(table(actual = y, predicted = knn_pred))

cat("Confusion Matrix: Decision Tree\n")
print(table(actual = y, predicted = dt_pred))

cat("Confusion Matrix: Logistic Regression\n")
print(table(actual = y, predicted = lr_pred))

cat("Confusion Matrix: Naive Bayes\n")
print(table(actual = y, predicted = nb_pred))

knn_tpr_final <- get_tpr(y, knn_pred)
dt_tpr_final  <- get_tpr(y, dt_pred)
lr_tpr_final  <- get_tpr(y, lr_pred)
nb_tpr_final  <- get_tpr(y, nb_pred)

knn_precision <- get_precision(y, knn_pred)
dt_precision  <- get_precision(y, dt_pred)
lr_precision  <- get_precision(y, lr_pred)
nb_precision  <- get_precision(y, nb_pred)

# =========================
# ROC AND AUC
# Topic 7 uses ROCR
# =========================
pred_knn <- prediction(knn_prob, y_num)
perf_knn <- performance(pred_knn, "tpr", "fpr")
auc_knn <- performance(pred_knn, "auc")@y.values[[1]]

pred_dt <- prediction(dt_prob, y_num)
perf_dt <- performance(pred_dt, "tpr", "fpr")
auc_dt <- performance(pred_dt, "auc")@y.values[[1]]

pred_lr <- prediction(lr_prob, y_num)
perf_lr <- performance(pred_lr, "tpr", "fpr")
auc_lr <- performance(pred_lr, "auc")@y.values[[1]]

pred_nb <- prediction(nb_prob, y_num)
perf_nb <- performance(pred_nb, "tpr", "fpr")
auc_nb <- performance(pred_nb, "auc")@y.values[[1]]

plot(perf_knn, col = "blue", lwd = 2, main = "ROC Curves for All Models")
plot(perf_dt, add = TRUE, col = "red", lwd = 2)
plot(perf_lr, add = TRUE, col = "darkgreen", lwd = 2)
plot(perf_nb, add = TRUE, col = "purple", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "grey")

legend("bottomright",
       legend = c("KNN", "Decision Tree", "Logistic Regression", "Naive Bayes"),
       col = c("blue", "red", "darkgreen", "purple"),
       lwd = 2,
       bty = "n")

# =========================
# FINAL COMPARISON TABLE
# =========================
results <- data.frame(
  Model = c("KNN", "Decision Tree", "Logistic Regression", "Naive Bayes"),
  TPR = c(knn_tpr_final, dt_tpr_final, lr_tpr_final, nb_tpr_final),
  Precision = c(knn_precision, dt_precision, lr_precision, nb_precision),
  AUC = c(auc_knn, auc_dt, auc_lr, auc_nb)
)

print(results)