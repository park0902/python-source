class InstanceVar:
    def __init__(self):
        self.text_list = []

    def add(self,text):
        self.text_list.append(text)

    def print_list(self):
        print(self.text_list)

if __name__ == '__main__':

    a = InstanceVar()
    a.add('a')
    a.print_list()

    b = InstanceVar()
    b.add('b')
    b.print_list()