class ContactInfo():
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def print_info(self):
        print('{0} : {1}'.format(self.name, self.email))

if __name__ == '__main__':
    sangboem = ContactInfo('박상범', 'jm050106@naver.com')

    sangboem.print_info()