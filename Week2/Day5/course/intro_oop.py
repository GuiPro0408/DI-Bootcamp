class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def show_details(self):
        print("Hello my name is " + self.name)

    def set_name(self, name):
        self.name = name


first_person = Person("John", 36)
first_person.show_details()

first_person.set_name("Jane")
first_person.show_details()
