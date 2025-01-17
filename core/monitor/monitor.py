from core.monitor.chrono import Chrono
from core.monitor.ram import RAM

class Monitor:
    
    """
    Meta-structure to monitor some variables in a script
    """
    
    def __init__(self, time=True, ram=True):
        self.trackers = []
        if ram : self.trackers.append(RAM())
        if time: self.trackers.append(Chrono())         
    
    def __enter__(self):
        for t in self.trackers:
            t.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for t in self.trackers:
            t.__exit__(exc_type, exc_val, exc_tb)
    
    def restart(self):
        for t in self.trackers:
            t.restart()
    
    def laps(self):
        for t in self.trackers:
            t.laps()
        
    def stop(self):
        for t in self.trackers:
            t.stop()

    def display(self):
        for t in self.trackers:
            t.display()