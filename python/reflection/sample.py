class Sample(object):
    """
    """
    
    def __init__(self, ):
        """
        """
        

    def add(self, x, y):
        """
        
        Arguments:
        - `x`:
        - `y`:
        """
        return x + y
        
    def multiply(self, x, y):
        """
        
        Arguments:
        - `x`:
        - `y`:
        """
        return x * y

if __name__ == '__main__':
    sample = Sample()
    if hasattr(sample, "add"):
        attr = getattr(sample, "add")
        print attr
        print attr(10, 20)
        
