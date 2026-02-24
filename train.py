import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():

    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=5000))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", accuracy)

    joblib.dump(pipe, "model.pkl")
    print("Pipeline saved successfully.")
    print("CI/CD Automation is Working! Accuracy:", accuracy)

if __name__ == "__main__":
    main()
