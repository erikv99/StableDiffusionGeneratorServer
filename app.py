from flask import Flask
from flask_restful import Api
from generator_api import GeneratorAPI

app = Flask(__name__)

api = Api(app)
api.add_resource(GeneratorAPI, '/generate')

def main():

    # TODO:  Alter before deployment
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
