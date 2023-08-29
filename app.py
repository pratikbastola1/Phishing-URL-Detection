from flask import Flask, request, render_template
import numpy as np
import warnings
import pickle
from feature import FeatureExtraction

# Turn off Python warnings
warnings.filterwarnings('ignore')

# Load the model only once when the application starts
with open("pickle/model.pkl", "rb") as file:
    gbc = pickle.load(file)

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, 30)

        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

        return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url)

    return render_template("index.html", xx=-1)


if __name__ == "__main__":
    # Remember to turn off debug mode when deploying to production
    app.run(debug=True)
ss