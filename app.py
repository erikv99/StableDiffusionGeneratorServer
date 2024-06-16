from flask import Flask
from flask_restful import Api

from api.clear_cache import ClearCache
from api.enqueue_generation import EnqueueGeneration
from generator import Generator

import os

app = Flask(__name__)
api = Api(app)

# This is temporary, input will be handled differently later.
def _validate_input_dir():

    if not os.path.exists("./input"):

        print("Input directory not found. Creating input directory.")
        os.makedirs("./input")

    if not os.path.exists("./input/default"):

        print("Default input directory not found. Creating default input directory.")
        os.makedirs("./input/default")

    # Todo: Add some default images to the default input directory i guess

def _validate_output_dir():

    if not os.path.exists("./output"):

        print("Output directory not found. Creating output directory.")
        os.makedirs("./output")

def main():
    
    _validate_output_dir()
    _validate_input_dir()
    generator = Generator()

    api.add_resource(EnqueueGeneration, '/enqueue-generation', resource_class_kwargs={'generator': generator})
    api.add_resource(ClearCache, '/clear-cache')

    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
