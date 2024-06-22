from flask import jsonify
from flask_restful import Resource

from generator import Generator

# Will attempt to retrieve the output from a specific generation.

class RetrieveOutput(Resource):
    
    def __init__(self, generator: Generator):
        self._generator = generator
        print("\nRetrieve Output API initialized.\n")
        
    def get(self, id):
        
        print("Retrieving output with id:", id)
        pass
        # TODO: Grab the image with that id from the storage we still have to implement.

    def get(self, id):
        
        result = self.generator.get_result(id)
        
        if result is None:
            return jsonify({'status': 'pending'}), 202
        
        return jsonify({'status': 'completed', 'result': result})