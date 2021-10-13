class List:


    def __init__(self, data=None):
        self.data = data
        self.next = None  # List

        
    def add(self, data):
        if self.data is None:
            self.data = data
            return
        this = self
        while True:
            if this.next is None:
                this.next = List(data)
                break
            this = this.next


    def show(self):
        if self.data is None:
            return
        
        this = self
        while True:
            print(this.data)
            if this.next is None:
                break
            this = this.next


def main():
    list_data = List()
    list_data.add(0)
    list_data.add(1)
    list_data.add(2)
    list_data.add(3)
    list_data.add(4)
    list_data.show()


if __name__ == '__main__':
    main()
