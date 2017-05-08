class ClassVar:
    text_list = []

    def add(self,text):
        self.text_list.append(text)

    def print_list(self):
        print(self.text_list)

if __name__ == '__main__':
    a = ClassVar()
    a.add('a')
    a.print_list()

    b = ClassVar()
    b.add('b')
    b.print_list()