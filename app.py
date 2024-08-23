from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

model = load_model('NN_model_50_inverse.h5')
encoder = joblib.load('encoder.pkl')

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/api', methods=['POST'])
def api():
  data = request.form
  print(data)
  return data

@app.route('/predict', methods=['POST'])
def predict():

  if 'file' not in request.files:
    return jsonify({'error': 'No file part in the request'}), 400
  
  file = request.files['file']
  
  if file.filename == '':
    return jsonify({'error': 'No file selected for uploading'}), 400
  
  if file and file.filename.endswith('.csv'):
    data = pd.read_csv(file, delimiter=',|;')

    try:
      data = data[['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'GRADE']]
    except:
      data = data[['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'CONT. GRADE']]
      data = data.rename(columns={'CONT. GRADE': 'GRADE'})
      

    encoded_cols = encoder.transform(data[['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'GRADE']])
    encoded_data_test = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'GRADE']))

    df_encoded = pd.concat([data.reset_index(drop=True), encoded_data_test], axis=1)
    df_encoded = df_encoded.drop(['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'GRADE'], axis=1).values
    
    predictions = model.predict(df_encoded)
    
    del data, encoded_cols, encoded_data_test, df_encoded

    return jsonify({'predictions': predictions.tolist()})
  else:
    return jsonify({'error': 'Invalid file format, only .csv allowed'}), 400

if __name__ == '__main__':
    app.run(
      # debug=True,
      # host="192.168.15.234"
    )