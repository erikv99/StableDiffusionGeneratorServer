import os
import queue
import threading
import time
import uuid

from generator import Generator

class QueueHandler:
    
    SAVE_TO_SERVER = True
    
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
                
                if (self.SAVE_TO_SERVER):
                    self._save_img_to_server(result, request_uuid)

                self.results[request_uuid] = result
                self.request_queue.task_done()
                print(f"Finished processing request #{request_uuid}")

            except queue.Empty:
                pass

            except Exception as e:
                print(f"An error occurred while processing request: {str(e)}")
                # Store the error in the results dict?, TODO: better error handling

    def enqueue_request(self, settings):
        request_uuid = str(uuid.uuid4())
        self.request_queue.put((request_uuid, settings))
        return request_uuid

    def get_result(self, request_uuid):
        return self.results.get(request_uuid)

    def clear_queue(self):
        with self.request_queue.mutex:
            self.request_queue.queue.clear()

    def _save_img_to_server(self, image, uuid):

        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        current_time = time.localtime()
        timestamp = time.strftime("%H%M_%d_%m_%Y", current_time)
        filename = f"{timestamp}_{uuid}.png"
        output_path = os.path.join(output_dir, filename)
        image.save(output_path)
