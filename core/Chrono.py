import time
from datetime import timedelta

class Chrono:
    """
    - string: str
    - unit "m" | "s" | "ms" | "us"
    """
    
    def __init__(self, string='chrono object', unit="m"):
        assert unit in ["m", "s", "ms", "us"]
        self.unit = unit
        self.paused = False
        self.string = string
        self.start_t: float = time.time()
        self.stop_t:  float = None
    
    def __enter__(self):
        self.restart()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.print(self.unit)
    
    def restart(self):
        self.paused = False
        self.start_t = time.time()
    
    def pause(self) -> timedelta:
        if self.paused:
            raise RuntimeError('Cannot pause already paused chrono object')
        self.paused = True
        self.stop_t = time.time()

    def elapsed(self):
        if self.paused:
            return timedelta(seconds=(self.stop_t - self.start_t))
        else:
            return timedelta(seconds=(time.time() - self.start_t))
        
    def reset(self):
        self.stop_t = time.time()
        dt = timedelta(seconds=(self.stop_t - self.start_t))
        self.start_t: float = time.time()
        self.stop_t:  float = None
    
    def laps(self):
        self.reset()
        
    def stop(self):
        self.stop_t = time.time()
        self.paused = True
        return timedelta(seconds=(self.stop_t - self.start_t))
    
    def isPaused(self):
        return self.paused

    def print(self, unit="m"):
        
        assert unit in ["m", "s", "ms", "us"]
        
        t = self.elapsed().total_seconds()
        if unit == "m":
            m = int(t) // 60
            s = int(t) % 60
            print(f"Chrono: [{self.string}] took {m:02}m{s:02}s to complete.")
        else:
            coefs = {
                "s":  1, 
                "ms": 1e3, 
                "us": 1e6
            }
            coef = coefs[unit]
            
            print(f"Chrono: [{self.string}] took {t * coef:.3f}{unit} to complete.")