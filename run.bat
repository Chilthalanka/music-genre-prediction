echo on

:: Create the virtual environment cwij_msc_env
echo Creating virtual environment...
python -m venv cwij_msc_env

:: Activate the virtual environment cwij_msc_env
echo Activating virtual environment...
call cwij_msc_env\Scripts\activate.bat

:: Install the required packages
echo Installing required packages...
pip install -r requirements.txt

:: Download NLTK data
echo Downloading NLTK data...
python -m nltk.downloader wordnet

:: Set PYSPARK_PYTHON environment variable
echo Set PYSPARK_PYTHON=python
set PYSPARK_PYTHON=python

:: Run the flask app
echo Starting Flask app...
python app.py
