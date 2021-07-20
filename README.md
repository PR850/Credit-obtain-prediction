# Credit obtain prediction

Program predict thought varius inputs if user get credit. Inputs can be changed in X_test.xls.

train_model.py has dwo methods: oneDense, twoDense which train model with one dense layer and two dense layers, for 1, 2, 4, 8, 16, 32, 64 neurons and save each model for later comparing.
Best model can be selected by using tensorboard.

predict.py gets data from X_test.xls and predict throu given model. Model have to be changed manualy by changing the path.
keras has to be inported even if it's unused due to tensorflow model management
