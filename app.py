from flask import Flask
from flask_restful import Api
from generator_api import GeneratorAPI
# from OpenSSL import SSL

app = Flask(__name__)

# CERT_FILE = "/path/to/cert.pem" 
# KEY_FILE = "/path/to/key.pem" 

# # Create SSL context 
# context = SSL.Context(SSL.PROTOCOL_TLSv1_2) 
# context.load_cert_chain(CERT_FILE, KEY_FILE) 

api = Api(app)
api.add_resource(GeneratorAPI, '/generate')

def main():

    # TODO:  Alter before deployment
    # app.run(debug=True, host='0.0.0.0', ssl_context=context)
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
