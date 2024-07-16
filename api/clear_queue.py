from flask_restful import Resource
from queue_handler import QueueHandler

class ClearQueue(Resource):
    
    def __init__(self, queue_handler: QueueHandler):
        self._queue_handler = queue_handler
        
    def get(self):
        self._queue_handler.clear_queue()
        return {"message": "Queue cleared."}