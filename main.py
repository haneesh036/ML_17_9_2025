import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

# A1. BASIC MODULES
def sum_unit(inputs_vec, weight_vec):
    return np.dot(inputs_vec, weight_vec)

def step_fn(val): return 1 if val >= 0 else 0
def bipolar_step_fn(val): return 1 if val >= 0 else -1
def sigmoid_fn(val): return 1 / (1 + np.exp(-val))
def relu_fn(val): return np.maximum(0, val)

def error_diff(pred_val, true_val): return true_val - pred_val

# A2. PERCEPTRON LEARNING
def perceptron_fit(features, labels, init_weights, learn_rate=0.05, act_type="step", max_iter=1000, tolerance=0.002):
    error_history = []
    epoch_count = 0
    act_lookup = {"step": step_fn, "bipolar": bipolar_step_fn,
                  "sigmoid": sigmoid_fn, "relu": relu_fn}
    act_func = act_lookup[act_type]

    while epoch_count < max_iter:
        epoch_error = 0
        for idx in range(len(features)):
            net_val = sum_unit(features[idx], init_weights)
            output = act_func(net_val)
            err = error_diff(output, labels[idx])
            init_weights = init_weights + learn_rate * err * features[idx]
            epoch_error += err**2
        error_history.append(epoch_error)
        if epoch_error <= tolerance:
            break
        epoch_count += 1
    return init_weights, error_history, epoch_count

# A8 & A9. BACKPROP (from scratch)
def backprop_and_gate(learn_rate=0.05, max_iter=1000, tolerance=0.002):
    feat_mat = np.array([[0,0],[0,1],[1,0],[1,1]])
    target_mat = np.array([[0],[0],[0],[1]])

    np.random.seed(42)
    w_hidden = np.random.randn(2,2); b_hidden = np.zeros((1,2))
    w_out = np.random.randn(2,1); b_out = np.zeros((1,1))
    mse_list = []

    for epoch in range(max_iter):
        z_hidden = feat_mat @ w_hidden + b_hidden; a_hidden = sigmoid_fn(z_hidden)
        z_out = a_hidden @ w_out + b_out; a_out = sigmoid_fn(z_out)

        err_mat = target_mat - a_out
        mse = np.mean(err_mat**2); mse_list.append(mse)
        if mse <= tolerance: break

        grad_out = err_mat * a_out * (1-a_out)
        dw_out = a_hidden.T @ grad_out; db_out = np.sum(grad_out, axis=0, keepdims=True)
        grad_hidden = (grad_out @ w_out.T) * a_hidden * (1-a_hidden)
        dw_hidden = feat_mat.T @ grad_hidden; db_hidden = np.sum(grad_hidden, axis=0, keepdims=True)

        w_hidden += learn_rate*dw_hidden; b_hidden += learn_rate*db_hidden
        w_out += learn_rate*dw_out; b_out += learn_rate*db_out

    return mse_list, np.round(a_out.ravel())

def backprop_xor_gate(learn_rate=0.05, max_iter=1000, tolerance=0.002):
    feat_mat = np.array([[0,0],[0,1],[1,0],[1,1]])
    target_mat = np.array([[0],[1],[1],[0]])

    np.random.seed(42)
    w_hidden = np.random.randn(2,4); b_hidden = np.zeros((1,4))
    w_out = np.random.randn(4,1); b_out = np.zeros((1,1))
    mse_list = []

    for epoch in range(max_iter):
        z_hidden = feat_mat @ w_hidden + b_hidden; a_hidden = sigmoid_fn(z_hidden)
        z_out = a_hidden @ w_out + b_out; a_out = sigmoid_fn(z_out)

        err_mat = target_mat - a_out
        mse = np.mean(err_mat**2); mse_list.append(mse)
        if mse <= tolerance: break

        grad_out = err_mat * a_out * (1-a_out)
        dw_out = a_hidden.T @ grad_out; db_out = np.sum(grad_out, axis=0, keepdims=True)
        grad_hidden = (grad_out @ w_out.T) * a_hidden * (1-a_hidden)
        dw_hidden = feat_mat.T @ grad_hidden; db_hidden = np.sum(grad_hidden, axis=0, keepdims=True)

        w_hidden += learn_rate*dw_hidden; b_hidden += learn_rate*db_hidden
        w_out += learn_rate*dw_out; b_out += learn_rate*db_out

    return mse_list, np.round(a_out.ravel())

# A10. TWO OUTPUT ENCODING
def mlp_dual_output():
    feat_mat = np.array([[0,0],[0,1],[1,0],[1,1]])
    target_and = np.array([[1,0],[1,0],[1,0],[0,1]])  
    target_xor = np.array([[1,0],[0,1],[0,1],[1,0]])  

    mlp_and = MLPClassifier(hidden_layer_sizes=(4,), activation="logistic",
                            solver="lbfgs", max_iter=5000, random_state=42)
    mlp_and.fit(feat_mat, target_and)

    mlp_xor = MLPClassifier(hidden_layer_sizes=(4,2), activation="logistic",
                            solver="lbfgs", max_iter=5000, random_state=42)
    mlp_xor.fit(feat_mat, target_xor)

    return mlp_and.predict(feat_mat), mlp_xor.predict(feat_mat)

# A11. sklearn MLPClassifier (Fixed)
def mlp_with_sklearn():
    feat_mat = np.array([[0,0],[0,1],[1,0],[1,1]])
    target_and = np.array([0,0,0,1])
    target_xor = np.array([0,1,1,0])

    mlp_and = MLPClassifier(hidden_layer_sizes=(4,), activation="logistic",
                            solver="lbfgs", max_iter=5000, random_state=42)
    mlp_and.fit(feat_mat, target_and)

    mlp_xor = MLPClassifier(hidden_layer_sizes=(4,2), activation="logistic",
                            solver="lbfgs", max_iter=5000, random_state=42)
    mlp_xor.fit(feat_mat, target_xor)

    return mlp_and.predict(feat_mat), mlp_xor.predict(feat_mat)

# A12. Dataset Loader
def load_classification_dataset(file_path, target_col=None):
    df_data = pd.read_csv(file_path)
    df_numeric = df_data.select_dtypes(include=[np.number])

    if target_col is None:
        features = df_numeric.iloc[:,:-1].values
        labels = df_numeric.iloc[:,-1].values
    else:
        features = df_numeric.drop(columns=[target_col]).values
        labels = df_data[target_col].values

    filler = SimpleImputer(strategy="mean")
    features = filler.fit_transform(features)

    if np.issubdtype(labels.dtype, np.floating):  
        labels = pd.qcut(labels, q=3, labels=False)

    if labels.dtype == object:
        labels = LabelEncoder().fit_transform(labels)

    return features, labels

# MAIN
if __name__ == "__main__":
    # A2
    feat_and = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    target_and = np.array([0,0,0,1])
    init_w = np.array([10, 0.2, -0.75])
    _, err_hist, _ = perceptron_fit(feat_and, target_and, init_w, learn_rate=0.05)
    plt.plot(err_hist); plt.title("A2 -> AND Gate Training"); plt.show()

    # A3
    for act in ["step", "bipolar", "sigmoid", "relu"]:
        _, _, epoch_num = perceptron_fit(feat_and, target_and, init_w, learn_rate=0.05, act_type=act)
        print(f"A3 -> Activation {act}, epochs {epoch_num}")

    # A4
    lr_range = np.arange(0.1, 1.1, 0.1); epoch_list = []
    for lr_val in lr_range:
        _, _, epoch_num = perceptron_fit(feat_and, target_and, init_w, learn_rate=lr_val)
        epoch_list.append(epoch_num)
    plt.plot(lr_range, epoch_list, marker="o"); plt.title("A4 -> Effect of LR"); plt.show()

    # A5
    feat_xor = np.array([[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
    target_xor = np.array([0,1,1,0])
    _, err_hist_xor, _ = perceptron_fit(feat_xor, target_xor, init_w, learn_rate=0.05)
    plt.plot(err_hist_xor); plt.title("A5 -> XOR (Single Perceptron)"); plt.show()

    # A6
    cust_data = {"Candies":[20,16,27,19,24,22,15,18,21,16],
                 "Mangoes":[6,3,6,1,4,1,4,4,1,2],
                 "Milk":[2,6,2,2,2,5,2,2,4,4],
                 "Payment":[386,289,393,110,280,167,271,274,148,198],
                 "HighValue":[1,1,1,0,1,0,1,1,0,0]}
    df_cust = pd.DataFrame(cust_data)
    X_cust = df_cust[["Candies","Mangoes","Milk","Payment"]].values
    y_cust = df_cust["HighValue"].values
    cust_clf = MLPClassifier(hidden_layer_sizes=(5,), activation="logistic", max_iter=500, random_state=42)
    cust_clf.fit(X_cust, y_cust)
    print("A6 -> Customer Predictions:", cust_clf.predict(X_cust))

    # A7
    X_aug = np.hstack([np.ones((X_cust.shape[0],1)), X_cust])
    w_pinv = np.linalg.pinv(X_aug) @ y_cust
    y_pred_pinv = np.round(sigmoid_fn(X_aug @ w_pinv))
    print("A7 -> Pseudo-Inverse Accuracy:", np.mean(y_pred_pinv == y_cust))

    # A8
    err_and_bp, preds_and_bp = backprop_and_gate()
    print("A8 -> AND predictions:", preds_and_bp)
    print("A8 -> Final Error:", err_and_bp[-1])

    # A9
    err_xor_bp, preds_xor_bp = backprop_xor_gate()
    print("A9 -> XOR predictions:", preds_xor_bp)
    print("A9 -> Final Error:", err_xor_bp[-1])

    # A10
    preds_and2, preds_xor2 = mlp_dual_output()
    print("A10 -> 2-output AND predictions:\n", preds_and2)
    print("A10 -> 2-output XOR predictions:\n", preds_xor2)

    # A11
    preds_and_sklearn, preds_xor_sklearn = mlp_with_sklearn()
    print("A11 -> sklearn MLP AND predictions:", preds_and_sklearn)
    print("A11 -> sklearn MLP XOR predictions:", preds_xor_sklearn)

    # A12
    for file in [
        "20231225_dfall_obs_data_and_spectral_features_revision1_n469.csv",
        "20240106_dfall_obs_data_and_cepstral_features_revision1_n469.csv"
    ]:
        print("\n=== Running A12 with MLPClassifier on:", file, "===")
        features, labels = load_classification_dataset(file)

        std_scaler = StandardScaler()
        features = std_scaler.fit_transform(features)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        model_clf = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                                  solver="adam", max_iter=2000, learning_rate_init=0.001,
                                  early_stopping=True, random_state=42)
        model_clf.fit(X_train, y_train)

        y_pred = model_clf.predict(X_test)
        print("Classification Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

