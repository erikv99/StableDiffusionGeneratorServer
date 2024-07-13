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


    # Old implementation from generator.py, for reference. TODO: Remove after testing.            
            
    #     def _process_queue(self):
        
    #     while True:
    #         while self._status in [self.GeneratorStatus.Initializing, self.GeneratorStatus.Generating]:
    #             print(f"Generator is {self._status.name}. Waiting...")
    #             time.sleep(10)
            
    #         try:
    #             self._status = self.GeneratorStatus.Generating
                
    #             request_uuid, data = self._request_queue.get(timeout=self.QUEUE_CHECK_INTERVAL_SECONDS)
    #             print(f"Processing request #{request_uuid}")
                
    #             result = self._process_request_with_retry(data)
    #             self._results[request_uuid] = result
                
    #             print(f"Finished processing request #{request_uuid}")
    #             self._request_queue.task_done()

    #         except queue.Empty:
    #             print("Queue is empty. Continuing to wait...")

    #         except Exception as e:
    #             print(f"An error occurred while processing request: {str(e)}")
    #             # Store the error in the results dict?, TODO: better error handling

    #         finally:
    #             self._status = self.GeneratorStatus.Available
    #             if not self.DEBUG:
    #                 torch.cuda.memory_summary(device=None, abbreviated=False)

    # def _process_request_with_retry(self, settings, max_retries=3):
        
    #     for retry in range(max_retries):
        
    #         try:
    #             if self.DEBUG:
    #                 print(f"Mock Processing request: {settings}")
    #                 return None
    
    #             else:
    #                 image = self.generate_image(settings)
    #                 return image
        
    #         except torch.cuda.OutOfMemoryError as e:
                
    #             print(f"Out of memory error: {e}. Retrying ({retry + 1}/{max_retries}) after clearing cache...")
    #             torch.cuda.empty_cache()
    #             time.sleep(10)
        
    #     print(f"Processing failed, the maximum of {max_retries} retries has been hit.")
    #     print("Generation failed.")
    #     return None
