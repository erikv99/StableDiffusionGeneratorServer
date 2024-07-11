import queue
import threading
import uuid

from generator import Generator

class QueueHandler:
    
    def __init__(self, generator: Generator):
        self.generator = generator
        self.request_queue = queue.Queue()
        self.results = {}
        self.thread = threading.Thread(target=self._process_queue, daemon=True)
        self.thread.start()

    def _process_queue(self):
        
        while True:
            try:
                request_uuid, settings = self.request_queue.get(timeout=20)
                result = self.generator.generate_image(settings)
                self.results[request_uuid] = result
                self.request_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"Error processing request: {e}")

    def enqueue_request(self, settings):
        request_uuid = str(uuid.uuid4())
        self.request_queue.put((request_uuid, settings))
        return request_uuid

    def get_result(self, request_uuid):
        return self.results.get(request_uuid)

    def clear_queue(self):
        with self.request_queue.mutex:
            self.request_queue.queue.clear()