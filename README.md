# Breast Cancer Predictor 
 
A Python-based web and command-line application to predict breast cancer using a Logistic Regression model. The project uses a 5-feature model trained on the Breast Cancer dataset and provides predictions as well as model evaluation via a confusion matrix plot. 
 
--- 
 
## Features 
- Predicts whether a tumor is **Benign** or **Malignant** using 5 key features: 
    - radius_mean 
    - texture_mean 
    - perimeter_mean 
    - area_mean 
    - smoothness_mean 
- Uses **StandardScaler** for feature scaling. 
- Logistic Regression model trained and saved for reuse. 
- Generates **accuracy and confusion matrix plot**. 
- Ready for integration with a Flask web app. 
 
--- 
 
## Installation 
1. Clone this repository: 
``bash 
git clone https://github.com/YOUR_USERNAME/Breast_Cancer_Predictor.git 
`` 
2. Navigate to the project folder: 
``bash 
cd Breast_Cancer_Predictor 
`` 
3. Create a virtual environment: 
``bash 
python -m venv venv 
`` 
4. Activate the virtual environment: 
- Windows: 
``bash 
venv\Scripts\activate 
`` 
- macOS/Linux: 
``bash 
source venv/bin/activate 
`` 
5. Install dependencies: 
``bash 
pip install -r requirements.txt 
`` 
 
--- 
 
## Usage 
1. Train the Model 
``bash 
python train_5feature_model.py 
`` 
2. Evaluate the Model 
``bash 
python model_accuracy_save_plot.py 
`` 
3. Web App (Optional) 
- Integrate logistic_model.pkl and scaler.pkl into a Flask app. 
- Create a form to input 5 features and display prediction (Benign or Malignant). 
 
--- 
 
## Project Structure 
``` 
Breast_Cancer_Predictor/ 
ÃÄ breast_cancer.csv           # Dataset 
ÃÄ train_5feature_model.py     # Script to train model and scaler 
ÃÄ model_accuracy_save_plot.py # Script to evaluate model 
ÃÄ logistic_model.pkl          # Saved Logistic Regression model 
ÃÄ scaler.pkl                  # Saved StandardScaler 
ÃÄ static/                     # Folder for static files (optional for Flask) 
ÃÄ templates/                  # Flask HTML templates (optional) 
ÀÄ README.md 
``` 
 
## License 
MIT License 
 
## Author 
**Yovinne Shadora** 
- GitHub: https://github.com/Yovinne-Shadora-data
- Email: yovinne.shadora21@gmail.com 
