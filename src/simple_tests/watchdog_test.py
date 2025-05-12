
# import time module, Observer, FileSystemEventHandler
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler



# This is a simple example of using watchdog to monitor a directory for file changes
# and print the file content when a file is created.


 
class OnMyWatch:
    # Set the directory on watch
    watchDirectiories = "src/simple_tests/files/"



    def __init__(self, watchDirectiories = watchDirectiories):
        self.observer = Observer()
        print("Wathcng directory: ", self.watchDirectiories)
 
    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDirectiories, recursive = True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")
 
        self.observer.join()
 
 
class Handler(FileSystemEventHandler):
 
    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None
 
        elif event.event_type == 'created':
            # Event is created, you can process it now
            print("Watchdog received created event - % s." % event.src_path)

            # Print the file content
            return event.src_path

             
 
if __name__ == '__main__':
    watch = OnMyWatch()
    watch.run()
