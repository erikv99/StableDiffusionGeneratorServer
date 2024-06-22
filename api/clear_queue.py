from flask_restful import Resource
from generator import Generator

class ClearQueue(Resource):
    
    def __init__(self, generator: Generator):
        self.generator = generator
        print("\nClear Queue API initialized.\n")
        
    def get(self):
        self.generator.clear_queue()
        return {"message": "Queue cleared."}