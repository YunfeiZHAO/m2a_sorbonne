import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from torch.utils.tensorboard import SummaryWriter

data = np.load("15_scenes_Xy.npz")
X = data["X"]
y = data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=8/10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=7/8, random_state=42)


decision_function_shapes = ['ovo', 'ovr']
kernels = ['linear', 'poly', 'rbf']
C_values = [0.01, 0.1, 1, 10, 50]


def hyper_parameters_selection():
    writer = SummaryWriter("SVM_hyper_parameters")
    for decision_function_shape in decision_function_shapes:
        for kernel in kernels:
            for c in C_values:
                print(f'Training SVM with  decision={decision_function_shape}, kernal:{kernel}, C={c}')
                # TODO
                clf = make_pipeline(StandardScaler(), SVC(decision_function_shape=decision_function_shape, kernel=kernel, C=c))
                # Fit on sub-train set
                # TODO
                clf.fit(X_train, y_train)

                # Evaluate on val set
                score = clf.score(X_val, y_val)

                # TODO

                writer.add_hparams({'decision_function_shapes': decision_function_shape,
                                    'kernel': kernel,
                                    'C': c},
                                   {'accuracy': score})
    writer.close()


def evaluation():
    clf = make_pipeline(StandardScaler(), SVC(decision_function_shape="ovo", kernel="rbf", C=1))
    # Fit on sub-train set
    # TODO
    X_train_total = np.concatenate((X_train, X_val))
    y_train_total = np.concatenate((y_train, y_val))

    clf.fit(X_train_total, y_train_total)

    # Evaluate on val set
    score = clf.score(X_test, y_test)
    print(f'evaluation score: {score}')