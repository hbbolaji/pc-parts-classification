import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from pcpartsclassifier.pipeline.prediction import PredictionPipeline

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)


@app.route('/', methods=["GET"])
@cross_origin()
def home():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictionRoute():
  classifier = PredictionPipeline()

  if request.method == 'POST':
    file = request.files['file']
    if file is None or file.filename == '':
      return jsonify({'error': 'no file'})
    if not allowed_file(file.filename):
      return jsonify({'error': 'format not supported'})
    
    try:
      image_bytes = file.read()
      classifier.transform_image(image_bytes=image_bytes)
      response = classifier.get_prediction()
      return jsonify(response)
    except Exception as e:
      raise e
  return jsonify({'cpu': 20})



if __name__ == '__main__':
  # clApp = ClientApp()
  app.run(host='0.0.0.0', port=8080)