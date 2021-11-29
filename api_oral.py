# Prevent ImportError w/ flask
import werkzeug
werkzeug.cached_property = werkzeug.utils.cached_property
from werkzeug.datastructures import FileStorage
# ML/Data processing
import tensorflow.keras as keras
import tensorflow as tf
# RESTful API packages
from flask_restplus import Api, Resource
from flask import Flask, jsonify
# Utility Functions
from util import oralmodel

application = app = Flask(__name__)
api = Api(app, version="1.0", title="ORAL DETECTION API", 
        description="Identifying if an image is oral or not")
ns = api.namespace(
    "ArtificialIntelligence", 
    description="Represents the image category by the AI."
)

# Use Flask-RESTPlus argparser to process user-uploaded images
arg_parser = api.parser()
arg_parser.add_argument('image', location='files',
                           type=FileStorage, required=True)



model = tf.keras.models.load_model('/var/oral/oralclassification/modelweight/new_oralmodel')
print("Loaded model from disk")
model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])


# Add the route to run inference
@ns.route("/prediction")
class CNNPrediction(Resource):
    """Takes in the image, to pass to the CNN"""
    @api.doc(parser=arg_parser, 
             description="Let the AI predict if its oral or not.")
    def post(self):
        # A: get the image
        image = oralmodel.get_image(arg_parser)
        # B: preprocess the image
        final_image = oralmodel.preprocess_image(image)
        # C: make the prediction
        prediction = oralmodel.predict_oral(model, final_image)
        # return the classification
        return jsonify(prediction)


if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=8080)

    # from waitress import serve
    # serve(app, port=8080)


