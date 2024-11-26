# standard library imports
# ...

# third party imports
# ...

# sub package imports
# ...


class _color:
    
    silenced = False
    
    def __init__(self, value):
        self.string = value

    def __str__(self):
        return self.string if not _color.silenced else ""

class rgb:
    purple      = _color('\033[95m')
    blue        = _color('\033[94m')
    cyan        = _color('\033[96m')
    green       = _color('\033[92m')
    orange      = _color('\033[93m')
    red         = _color('\033[91m')
    bold        = _color('\033[1m')
    underline   = _color('\033[4m')
    default     = _color('\033[0m')
    
    __silenced = False
    
    def silence():
        """
        Disable all color variables by replacing them with an empty string
        """
        _color.silenced = True

    def restore():
        """
        Restore all the default colors of the class, opposite of silence
        """
        _color.silenced = False
            

def disp(*args):
    
    print(str(rgb.orange) + "WARNING Deprecated interface \'core.output.disp\', use core.log module instead" + str(rgb.default))

    message = ""
    for arg in args:
         message += str(arg)
    print(message + str(rgb.default))
    
def error(*args):
     
    print(str(rgb.orange) + "WARNING Deprecated interface \'core.output.error\', will soon be deleted" + str(rgb.default))
     
    disp(rgb.red, 'Error: ', *args)
    disp(rgb.orange, '> Aborting.', str(rgb.default))
    exit(-1)

def iferror(err, *args):
    
    print(str(rgb.orange) + "WARNING Deprecated interface \'core.output.iferror\', will soon be deleted" + str(rgb.default))
    
    assert type(err) is int
    
    if err != 0:
        error(*args)
        
print(str(rgb.orange) + "WARNING Deprecated module \'output\', use core.log and core.rgb module instead" + str(rgb.default))