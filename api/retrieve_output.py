import io
from flask import jsonify, send_file
from flask_restful import Resource

from queue_handler import QueueHandler

class RetrieveOutput(Resource):
    
    def __init__(self, queue_handler: QueueHandler):
        
        self.queue_handler = queue_handler
        print("\nRetrieve Output API initialized.\n")
        
    # def get(self, id):
        
    #     print("Retrieving output with id:", id)
    #     pass
    #     # TODO: Grab the image with that id from the storage we still have to implement.

    def get(self, id):
        
        image = self.queue_handler.get_result(id)
        
        if image is None:
            return jsonify({'status': 'pending'}), 202
        
        # TODO: Check if actually needed
        image.format = "PNG" 
        
        # Convert the image to bytes for sending it back
        bytes_io = io.BytesIO()
        image.save(bytes_io, 'PNG')
        bytes_io.seek(0)

        return send_file(bytes_io, mimetype='image/png')
        # return jsonify({'status': 'completed', 'result': image})