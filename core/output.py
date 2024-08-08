# standard library imports
# ...

# third party imports
# ...

# sub package imports
# ...

class rgb:
    purple = '\033[95m'
    blue = '\033[94m'
    cyan = '\033[96m'
    green = '\033[92m'
    orange = '\033[93m'
    red = '\033[91m'
    bold = '\033[1m'
    underline = '\033[4m'
    default = '\033[0m'

def disp(*args):

    message = ""
    for arg in args:
         message += arg
    print(message + rgb.default)
    
def error(*args):
     disp(rgb.red, 'Error: ', *args)
     disp(rgb.orange, '> Aborting.', rgb.default)
     exit(-1)

def iferror(err, *args):
    
    assert type(err) is int
    
    if err != 0:
        error(*args)