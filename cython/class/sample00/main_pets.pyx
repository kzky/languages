import pyximport; pyximport.install()
from pets import *

def main():

    parrot = Parrot()
    parrot.describe()

    norwegian = Norwegian()
    norwegian.describe()
    
if __name__ == '__main__':
    main()
