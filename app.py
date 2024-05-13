from flask import Flask
from flask_restful import Api
from generator_api import GeneratorAPI
from generator import Generator

app = Flask(__name__)

api = Api(app)
api.add_resource(GeneratorAPI, '/generate')

def main():

    # TODO:  Alter before deployment
    app.run(port='8080', debug=True)

if __name__ == "__main__":
    main()