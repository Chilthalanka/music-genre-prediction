@echo off

:: Create the virtual environment cwij_msc_env
echo Creating virtual environment...

python -m venv cwij_msc_env

:: Activate the virtual environment cwij_msc_env
echo Activating virtual environment...

cwij_msc_env\Scripts\activate

:: Install the required packages
echo Installing required packages...

pip install -r requirements.txt

:: Run the flask app
echo Starting Flask app...

python app.py
