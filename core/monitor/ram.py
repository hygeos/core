import tracemalloc

from core import log

class RAM:
    """
    - name: str
    """
    
    def __init__(self, name='ram object'):
        self.name = name
        self.paused = False
        self.peak: float = 0.
        self.current: float = 0.
        tracemalloc.start()
    
    def __enter__(self):
        self.restart()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.display()
    
    def restart(self):
        self.paused = False
        tracemalloc.start()
    
    def pause(self):
        if self.paused:
            raise RuntimeError('Cannot pause already paused chrono object')
        self.paused = True
        current, peak = self.stop()
        self.peak += peak
        self.current += current
 
    def elapsed(self):
        if not self.paused: 
            self.stop()
            self.restart()
        return self.peak, self.current
               
    def reset(self):
        current, peak = self.stop()
        self.restart()  
        self.peak: float = 0.
        self.current: float = 0.      
        return current, peak
    
    def laps(self):
        return self.reset()
        
    def stop(self):
        current, peak = tracemalloc.get_traced_memory()
        self.peak += peak
        self.current += current
        tracemalloc.stop()
        return self.current, self.peak

    def display(self):
        str_peak = self._format_size(self.peak)
        str_current = self._format_size(self.current)
        msg = "RAM usage: [{}] use {} at peak and {} at the end of execution."         
        log.info(msg.format(self.name, str_peak, str_current))
    
    def _format_size(self, byte_size):
        scale = {
            3: "KB",  
            6: "MB", 
            9: "GB", 
            12: "TB", 
        }
        
        sc = None
        if 0 <= byte_size < 10**3: 
            sc = 3
        elif 10**3 <= byte_size < 10**6: 
            sc = 3
        elif 10**6 <= byte_size < 10**9: 
            sc = 6
        elif 10**9 <= byte_size < 10**12: 
            sc = 9
        elif 10**12 <= byte_size:
            sc = 12
        else:
            raise Exception('ERROR ', f"byte_size: {byte_size}")
            
        number = byte_size / (1024**(sc/3))
        unit = scale[sc]
        
        return f"{number:0.1f}{unit}"