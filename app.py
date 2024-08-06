from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re

app = Flask(__name__)
CORS(app)

model = joblib.load('XGBoost_model_6depth_50(2).pkl')
# model = XGBRegressor()
# model.load_model('XGBoost_model_6depth_50(2).json')
encoder = joblib.load('encoder.pkl')

@app.route('/')
def index():
    return 'Hello, World!'

@app.route('/api', methods=['POST'])
def api():
  data = request.form
  print(data)
  return data

# @app.route('/predict', methods=['POST'])
# def predict():

#   if 'file' not in request.files:
#     return jsonify({'error': 'No file part in the request'}), 400
  
#   file = request.files['file']
  
#   if file.filename == '':
#     return jsonify({'error': 'No file selected for uploading'}), 400
  
#   if file and file.filename.endswith('.xlsx'):
#     new_data = pd.read_excel(file, skiprows=6)

#     cabang = pd.read_excel(file, skiprows=1).iloc[0, 2]
#     cabang = cabang.split()[0]
    
#     # new_data = new_data[(new_data['MILIK'] == 'COC') &
#     #                     (new_data['PREV. TO PREV. STATUS'] == 'FXD') &
#     #                     (new_data['CURR. STATUS'] == 'MTA')]
#     new_data = new_data[['CONSIGNEE', 'CONTAINER', 'CARGO']]
#     new_data['CABANG'] = cabang

#     new_data['CONTAINER'] = new_data['CONTAINER'].str.replace('SPNU', '')
#     new_data['CONTAINER'] = new_data['CONTAINER'].str[:3]
#     new_data['CONTAINER'] = pd.to_numeric(new_data['CONTAINER'], errors='coerce')
#     new_data['GRADE'] = new_data['CONTAINER'].apply(assign_grade)
#     new_data['SIZE'] = new_data['CONTAINER'].apply(assign_size)
#     new_data = new_data.drop(['CONTAINER'], axis=1)

    
#     encoded_cols = encoder.transform(new_data[['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'GRADE']])
#     encoded_data_test = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'GRADE']))
#     new_data_encoded = pd.concat([new_data.reset_index(drop=True), encoded_data_test], axis=1)
#     new_data_encoded = new_data_encoded.drop(['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'GRADE'], axis=1)
    
#     predictions = model.predict(new_data_encoded)
    
#     return jsonify({'predictions': predictions.tolist()})
#     # return jsonify({'anu': cabang})
#   else:
#     return jsonify({'error': 'Invalid file format, only .xlsx allowed'}), 400

@app.route('/predict', methods=['POST'])
def predict():

  if 'file' not in request.files:
    return jsonify({'error': 'No file part in the request'}), 400
  
  file = request.files['file']
  
  if file.filename == '':
    return jsonify({'error': 'No file selected for uploading'}), 400
  
  if file and file.filename.endswith('.xlsx'):
    new_data = pd.read_excel(file, skiprows=6)

    cabang = pd.read_excel(file, skiprows=1).iloc[0, 2]
    cabang = cabang.split()[0]
    
    # new_data = new_data[(new_data['MILIK'] == 'COC') &
    #                     (new_data['PREV. TO PREV. STATUS'] == 'FXD') &
    #                     (new_data['CURR. STATUS'] == 'MTA')]
    new_data = new_data[['CONSIGNEE', 'CONTAINER', 'CARGO']]
    new_data['CABANG'] = cabang

    new_data['CONTAINER'] = new_data['CONTAINER'].str.replace('SPNU', '')
    new_data['CONTAINER'] = new_data['CONTAINER'].str[:3]
    new_data['CONTAINER'] = pd.to_numeric(new_data['CONTAINER'], errors='coerce')
    new_data['GRADE'] = new_data['CONTAINER'].apply(assign_grade)
    new_data['SIZE'] = new_data['CONTAINER'].apply(assign_size)
    new_data = new_data.drop(['CONTAINER'], axis=1)

    
    encoded_cols = encoder.transform(new_data[['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'GRADE']])
    encoded_data_test = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'GRADE']))
    new_data_encoded = pd.concat([new_data.reset_index(drop=True), encoded_data_test], axis=1)
    new_data_encoded = new_data_encoded.drop(['CONSIGNEE', 'SIZE', 'CARGO', 'CABANG', 'GRADE'], axis=1)

    regex = re.compile(r"\[|\]|<", re.IGNORECASE)
    new_data_encoded.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in new_data_encoded.columns.values]
    
    predictions = model.predict(new_data_encoded)
    
    return jsonify({'predictions': predictions.tolist()})
    # return jsonify({'anu': cabang})
  else:
    return jsonify({'error': 'Invalid file format, only .xlsx allowed'}), 400
    
def assign_grade(container):
  if 290 <= container <= 400:
    return 'A'
  elif 280 <= container <= 289:
    return 'B'
  elif container <= 279:
    return 'C'
  elif container >= 463:
    return 'A'
  elif 461 <= container <= 462:
    return 'B'
  elif container == 460:
    return 'C'
  else:
    return 'Unknown'

def assign_size(container):
  if container <= 400:
    return 20
  else:
    return 40

if __name__ == '__main__':
    app.run(
      debug=True,
      # host="192.168.15.234"
    )