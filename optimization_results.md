# Model Optimization Results

To answer your question, **yes!** We can definitely improve the accuracy by applying optimization techniques. I took the best performing model from our previous tests—**LightGBM on the new dataset**—and applied a hyperparameter tuning technique called `RandomizedSearchCV`. 

I tested 100 different configurations using a 5-Fold Cross Validation pipeline, and the results showed a noticeable improvement!

## 🚀 Performance Leap

| Metric | Before Optimization | After Optimization | Improvement |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 83.91% | **86.08%** | 📈 **+ 2.17%** |
| **F1 Score** | 0.8148 | **0.8429** | 📈 **+ 0.0281** |

## ⚙️ The Winning Hyperparameters
The `RandomizedSearchCV` algorithm found that the LightGBM model performs best under these exact conditions:

* **n_estimators**: 500 *(more trees)*
* **learning_rate**: 0.1
* **max_depth**: 5 *(limits tree depth to prevent overfitting)*
* **num_leaves**: 127
* **min_child_samples**: 30 *(requires more samples in leaves, increasing robustness)*
* **subsample**: 0.6 *(uses 60% of data per tree, acting as regularization)*
* **colsample_bytree**: 1.0 *(uses all features per tree)*
* **reg_lambda**: 1.0 *(adds L2 regularization penalty to weights)*
* **reg_alpha**: 0.0

### Summary
By tuning the hyperparameters to regularize the tree (using L2 penalty `reg_lambda` and sample bagging `subsample`) and increasing the number of estimators, the model was able to generalize much better to the unseen validation folds, boosting your accuracy all the way up to **86.08%**! 
