import numpy as np
import os
from keras.models import load_model


def main():
    model_name = '12312313231'
    model_path = './logs/12312312312/' + model_name
    test_path = os.path.abspath('./test.csv')
    output_path = os.path.abspath('./' + model_name + '_ans.csv')
    test_arr = np.genfromtxt(test_path, delimiter=',', dtype='str', skip_header=1)

    X = np.zeros((test_arr.shape[0], 28*28))
    for i in range(0, test_arr.shape[0]):
        X[i][:] = test_arr[i][:]

    X = X.reshape((test_arr.shape[0], 28, 28, 1))
    X = X/255.0

    classifier = load_model(model_path)
    print(classifier.summary())
    ans_proba = classifier.predict(X, batch_size=256)
    ans_class = ans_proba.argmax(axis=-1)

    with open(output_path, 'w+', encoding='utf-8') as f:
        f.write('id,label' + '\n')
        for i in range(ans_class.shape[0]):
            f.write(str(i) + ',' + str(int(ans_class[i])) + '\n')
        f.close()
