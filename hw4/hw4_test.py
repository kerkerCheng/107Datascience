import numpy as np
import os
from keras.models import load_model
from scipy import stats

def main():
    model_name1 = 'model_2018.12.05_21.44_71_0.95.hdf5'
    model_path1 = model_name1

    model_name2 = 'model_2018.12.06_15.46_67_0.95.hdf5'
    model_path2 = model_name2

    model_name3 = 'model_2018.12.07_09.58_70_0.95.hdf5'
    model_path3 = model_name3

    test_path = os.path.abspath('./test.csv')
    output_path = os.path.abspath('./' + model_name1 + '--' + model_name2 + '--' + model_name3 + '_ans.csv')
    test_arr = np.genfromtxt(test_path, delimiter=',', dtype='str', skip_header=1)

    X = np.zeros((test_arr.shape[0], 28*28))
    for i in range(0, test_arr.shape[0]):
        X[i][:] = test_arr[i][1:]

    X = X.reshape((test_arr.shape[0], 28, 28, 1))
    X = X/255.0
    num_of_test = test_arr.shape[0]

    classifier1 = load_model(model_path1)
    classifier2 = load_model(model_path2)
    classifier3 = load_model(model_path3)

    ans_proba1 = classifier1.predict(X, batch_size=256, verbose=1)
    ans_class1 = ans_proba1.argmax(axis=-1)

    ans_proba2 = classifier2.predict(X, batch_size=256, verbose=1)
    ans_class2 = ans_proba2.argmax(axis=-1)

    ans_proba3 = classifier3.predict(X, batch_size=256, verbose=1)
    ans_class3 = ans_proba3.argmax(axis=-1)

    ans_class = []
    for i in range(num_of_test):
        nominee = [ans_class1[i], ans_class2[i], ans_class3[i]]
        ans = stats.mode(nominee)[0][0]
        ans_class.append(ans)

    with open(output_path, 'w+', encoding='utf-8') as f:
        f.write('id,label' + '\n')
        for i in range(len(ans_class)):
            f.write(str(i) + ',' + str(int(ans_class[i])) + '\n')
        f.close()


if __name__ == "__main__":
    main()
