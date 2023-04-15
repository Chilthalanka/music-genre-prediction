# Import libraries
import io
import base64
from flask import Flask, render_template, request
from pyspark.sql import SparkSession
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.tuning import CrossValidatorModel
import matplotlib.pyplot as plt


app = Flask(__name__)


# Initialize Spark session
spark = SparkSession\
    .builder\
    .appName("Music Genre Prediction")\
    .config("spark.driver.memory", "4g")\
    .config("spark.executor.memory", "4g")\
    .getOrCreate()


# Load the data prep, string indexer and ML models
data_prep_model = PipelineModel.load("./model/8-classes/data_prep/")
lr_model = CrossValidatorModel.load("./model/8-classes/logistic_regression")

# Label map
labels = ["pop", "country", "blues", "rock", "jazz", "reggae", "hip hop", "retro"]
label_map = {0:"pop", 1:"country", 2:"blues", 3:"rock", 4:"jazz", 5:"reggae", 6:"hip hop", 7:"retro"}


@app.route('/')
def home():
    """
    Home page
    """
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict the genre of user input lyrics
    """
    # Fetch user input
    lyrics = request.form['lyrics']
    input_df = spark.createDataFrame([("unknown", "unknown", "unknown", "unknown", str(lyrics))],
                                     ["artist_name", "track_name", "release_date", "genre", "lyrics"])

    # Data pre-processing
    preprocessed_df = data_prep_model.transform(input_df)

    # Model prediction
    y_pred = lr_model.transform(preprocessed_df)
    predicted_genre = label_map[int(y_pred.collect()[0]['prediction'])]
    prediction_probs = y_pred.collect()[0]["probability"]

    # Visualize the predictions
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(prediction_probs, labels=labels, startangle=90, autopct='%1.2f%%')
    ax.axis('equal')

    # Save the chart to a buffer
    buffer_ = io.BytesIO()
    plt.savefig(buffer_, format='png')
    buffer_.seek(0)

    # Encode the buffer in base64
    chart_url = base64.b64encode(buffer_.getvalue()).decode()
    plt.close(fig)

    return render_template('results.html', lyrics=lyrics,
                           prediction=predicted_genre, chart_url=chart_url)


if __name__ == '__main__':
    app.run(debug=True)
