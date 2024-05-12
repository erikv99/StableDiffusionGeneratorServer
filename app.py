from flask import Flask
from flask_restful import Api
from generatorApi import GeneratorAPI

app = Flask(__name__)

api = Api(app)
api.add_resource(GeneratorAPI, '/generate')

def main():

    # setting up the generator
    generator = Generator()

    # TODO:  Alter before deployment
    app.run(port='8080', debug=True)

if __name__ == "__main__":
    main()