import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Data Preparation
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 2. Train Models
model_a = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
model_a.fit(X_train, y_train)

model_b = LogisticRegression(max_iter=10000, C=0.5, random_state=42)
model_b.fit(X_train, y_train)

acc_a = accuracy_score(y_test, model_a.predict(X_test))
acc_b = accuracy_score(y_test, model_b.predict(X_test))


# 3. SHAP values
explainer_a = shap.TreeExplainer(model_a)
shap_values_a = explainer_a.shap_values(X_test)
vals_a = shap_values_a[1] if isinstance(shap_values_a, list) else shap_values_a

explainer_b = shap.LinearExplainer(model_b, X_train)
vals_b = explainer_b.shap_values(X_test)


# 4. Plots (using SHAP summary plots)
print("Displaying Model A Explanation... (Close the plot to continue)")
# --- MODEL A ---
shap.summary_plot(vals_a, X_test, plot_type="bar", show=False, max_display=10)

figA = plt.gcf()  # get figure created by SHAP
figA.suptitle(f"MODEL A: Random Forest\n(Accuracy: {acc_a:.2f})")
plt.tight_layout()
plt.show()

print("Displaying Model B Explanation...")
# --- MODEL B ---
shap.summary_plot(vals_b, X_test, plot_type="bar", show=False, max_display=10)

figB = plt.gcf()  # get figure created by SHAP
figB.suptitle(f"MODEL B: Logistic Regression\n(Accuracy: {acc_b:.2f})")
plt.tight_layout()
plt.show()


# 5. Print top 5 important features for each model
def top_features(shap_values, X, k=5):
    # Convert to numpy array
    shap_vals = np.array(shap_values)

    # If SHAP returns 3D (TreeExplainer sometimes returns [samples, features, 1])
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0]

    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    idx = np.argsort(mean_abs)[::-1][:k]
    cols = X.columns.to_numpy()

    return [(cols[i], float(mean_abs[i])) for i in idx]

def features_to_df(model_name, feats):
    df = pd.DataFrame(feats, columns=["Feature", f"{model_name} Importance"])
    return df

def print_centered_table(df):
    col1 = df.columns[0]
    col2 = df.columns[1]
    print(f"{col1:^25} | {col2:^30}")
    print("-" * 60)

    for f, v in df.values:
        print(f"{f:^25} | {v:^30.6f}")

df_rf = features_to_df("Random Forest", top_features(vals_a, X_test))
df_lr = features_to_df("Logistic Regression", top_features(vals_b, X_test))

print("\n\t[Top 5 Important Features (Random Forest)]")
print_centered_table(df_rf)

print("\n\n\t[Top 5 Important Features (Logistic Regression)]")
print_centered_table(df_lr)