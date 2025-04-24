import sys, os
from yoloapp.utils.main_utils import decodeImage, encodeImageIntoBase64
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"

clApp = ClientApp()

@app.route("/")
def home():
    return render_template("index.html")

import glob

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    try:
        image_file = request.files['file']
        image_file.save(clApp.filename)

        # Run YOLOv5 detection
        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source ../inputImage.jpg")

        # Find the latest exp folder
        exp_dirs = sorted(glob.glob("yolov5/runs/detect/exp*"), key=os.path.getmtime, reverse=True)
        if not exp_dirs:
            raise FileNotFoundError("No YOLO output directory found.")
        
        # Get image path from latest exp folder
        detected_image_path = os.path.join(exp_dirs[0], "inputImage.jpg")
        if not os.path.exists(detected_image_path):
            # Try to find any image file if YOLO renamed it
            image_files = glob.glob(os.path.join(exp_dirs[0], "*.jpg"))
            if not image_files:
                raise FileNotFoundError("No output image found in YOLO folder.")
            detected_image_path = image_files[0]

        opencodedbase64 = encodeImageIntoBase64(detected_image_path)

        # Clean up
        os.system("rm -rf yolov5/runs")

        return jsonify({
            "image": opencodedbase64.decode('utf-8'),
            "detection": "wildlife Detected"
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500



    except ValueError as val:
        print(val)
        return Response("Value not found inside JSON data", status=400)
    except KeyError:
        return Response("Key value error: incorrect key passed", status=400)
    except Exception as e:
        print("Exception occurred:", e)
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route("/live", methods=['GET'])
@cross_origin()
def predictLive():
    try:
        os.system("cd yolov5/ && python detect.py --weights best.pt --img 416 --conf 0.5 --source 0")
        os.system("rm -rf yolov5/runs")
        return "Camera started!"
    except ValueError as val:
        print(val)
        return Response("Value not found inside JSON data", status=400)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
