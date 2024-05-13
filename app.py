from flask import Flask
from flask_restful import Api
from generatorApi import GeneratorAPI
from generator import Generator

app = Flask(__name__)

api = Api(app)
api.add_resource(GeneratorAPI, '/generate')

def main():

    # Setting up the generator, since i test on a NVIDIA RTX A4000 this will only use a single generation.
    # Multi generator support will be added later if needed.
    generator = Generator()

    # TODO:  Alter before deployment
    app.run(port='8080', debug=True)

if __name__ == "__main__":
    main()