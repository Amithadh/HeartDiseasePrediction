from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("dataset.csv")
y = dataset["target"]
X = dataset.drop(["target"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        age = request.form.get("age")
        gender = request.form.get("gender")
        cp = request.form.get("cp")
        trestbps = request.form.get("TRestBPS")
        chol = request.form.get("chol")
        fbs = request.form.get("fbs")
        restecg = request.form.get("RestECG")
        thalach = request.form.get("thalach")
        exang = request.form.get("exang")
        oldpeak = request.form.get("oldpeak")
        slope = request.form.get("slope")
        ca = request.form.get("ca")
        thal = request.form.get("thal")
        global knn
        res = knn.predict(
            np.array(
                [
                    [
                        age,
                        gender,
                        cp,
                        trestbps,
                        chol,
                        fbs,
                        restecg,
                        thalach,
                        exang,
                        oldpeak,
                        slope,
                        ca,
                        thal,
                    ]
                ],
                dtype=np.float32,
            )
        )
        print(res)
        return render_template("index.html", res=res[0])

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
