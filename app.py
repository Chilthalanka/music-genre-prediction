# Import libraries
import io
import base64
import os
from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.tuning import CrossValidatorModel
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


app = Flask(__name__)


# Initialize Spark session
spark = SparkSession\
    .builder\
    .appName("Music Genre Prediction")\
    .config("spark.driver.memory", "4g")\
    .config("spark.executor.memory", "4g")\
    .getOrCreate()


# Load the data prep, string indexer and ML models
model_base_path = "./model/8-classes/word2vec_only_v2/"
data_prep_model = PipelineModel.load(os.path.join(model_base_path, "data_prep/"))
lr_model = CrossValidatorModel.load(os.path.join(model_base_path, "logistic_regression/"))
rf_model = CrossValidatorModel.load(os.path.join(model_base_path, "random_forest/"))

# Label map
labels = ["pop", "country", "blues", "rock", "jazz", "reggae", "hip hop", "retro"]
label_map = {0:"pop", 1:"country", 2:"blues", 3:"rock",
             4:"jazz", 5:"reggae", 6:"hip hop", 7:"retro"}


@app.route('/')
def home():
    """
        Home page
    """
    return render_template("home.html")


@app.route('/results', methods=['POST'])
def results():
    """
        Predict the genre of user input lyrics
    """
    # Fetch user input
    lyrics = request.form['lyrics']
    model_name = request.form['model']

    # Get the relavant model
    if model_name=="logistic_regression":
        model = lr_model
    elif model_name=="random_forest":
        model = rf_model

    input_df = spark.createDataFrame([("unknown", "unknown", "unknown", "unknown", str(lyrics))],
                                    ["artist_name", "track_name", "release_date", "genre", "lyrics"])

    # Data pre-processing
    preprocessed_df = data_prep_model.transform(input_df)

    # Model prediction
    y_pred = model.transform(preprocessed_df)
    predicted_genre = label_map[int(y_pred.collect()[0]['prediction'])]
    prediction_probs = y_pred.collect()[0]["probability"]

    # Visualize the predictions
    # Create pie chart of class probabilities
    fig1, ax1 = plt.subplots()
    ax1.pie(prediction_probs, labels=labels, autopct='%1.1f%%')
    ax1.set_title('Class Probabilities')
    pie_chart = plot_to_img(fig1)
    plt.close(fig1)

    # Create bar chart of class probabilities
    fig2, ax2 = plt.subplots()
    ax2.bar(x=labels, height=prediction_probs)
    ax2.set_title('Class Probabilities')
    ax2.set_xlabel('Genre')
    ax2.set_ylabel('Probability')
    plt.xticks(rotation=45)
    bar_chart = plot_to_img(fig2)
    plt.close(fig2)

    return render_template('results.html',
                           lyrics=lyrics,
                           prediction=predicted_genre,
                           pie_chart=pie_chart,
                           bar_chart=bar_chart)


@app.route('/home')
def go_home():
    """
        Return to home page
    """
    return render_template('home.html')


def plot_to_img(plot):
    """
        Helper function to convert plot to image
    """
    img = io.BytesIO()
    plot.savefig(img, format='png')
    img.seek(0)
    img = base64.b64encode(img.getvalue()).decode()
    return "data:image/png;base64,{}".format(img)


if __name__ == '__main__':
    app.run(debug=True)
