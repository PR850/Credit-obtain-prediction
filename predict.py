import pandas as pd
import keras
import tensorflow as tf

model = tf.keras.models.load_model('Kredyt_20e_relu_sigm_adam_bin_crossen_Dense_8')

df = pd.read_excel('X_test.xls')
maxmin = pd.read_excel('maxmin.xls')

max = maxmin.iloc[0, 0:]
min = maxmin.iloc[1, 0:]

X_test = df.iloc[0, :]

x_list = X_test.values.tolist()
min_list = min.values.tolist()
max_list = max.values.tolist()

for i in range(20):
    maxx = max_list[i]
    minn = min_list[i]

    if x_list[i] > max_list[i]:
        maxx = x_list[i]
    if x_list[i] < min_list[i]:
        minn = x_list[i]

    norm = (x_list[i] - minn) / (maxx - minn)
    x_list[i] = norm

x_predict = pd.DataFrame(x_list)
x_predict = x_predict.transpose()
x_predict.to_csv('normalized_test.csv', header=None, index=False, sep=',')

x_predict = pd.read_csv('normalized_test.csv', header=None)

prediction = float(model.predict(x_predict)[0][0]) * 100

print("Szansa na dostanie kredytu: ", str(prediction)[:5], "%")
print("Credit Rating: 1") if prediction > 80 else print("Credit Rating: 0")
