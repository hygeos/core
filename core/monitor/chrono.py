import time
from datetime import timedelta

from core import log

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
        self.total_t: float = 0.
    
    def __enter__(self):
        self.restart()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.display(self.unit)
    
    def restart(self):
        self.paused = False
        self.start_t = time.time()
    
    def pause(self):
        if self.paused:
            raise RuntimeError('Cannot pause already paused chrono object')
        self.paused = True
        self.total_t += time.time()- self.start_t

    def elapsed(self) -> timedelta:
        if self.paused:
            return timedelta(seconds=(self.total_t))
        else:
            return timedelta(seconds=(time.time() - self.start_t))
        
    def reset(self) -> timedelta:
        dt = self.elapsed()
        self.start_t: float = time.time()
        self.total_t: float = 0.
        return dt
    
    def laps(self) -> timedelta:
        return self.reset()
        
    def stop(self) -> timedelta:
        self.paused = True
        self.total_t += time.time() - self.start_t
        return timedelta(seconds=self.total_t)

    def display(self, unit="m"):
        
        assert unit in ["m", "s", "ms", "us"]
        
        t = self.elapsed().total_seconds()
        if unit == "m":
            m = int(t) // 60
            s = int(t) % 60
            log.info(f"Chrono: [{self.string}] took {m:02}m{s:02}s to complete.")
        else:
            coefs = {
                "s":  1, 
                "ms": 1e3, 
                "us": 1e6
            }
            coef = coefs[unit]
            
            log.info(f"Chrono: [{self.string}] took {t * coef:.3f}{unit} to complete.")