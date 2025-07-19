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


# Try to recreate the class explained below:
#
# We have a class called Door that has an attribute of is_opened declared when an instance is initiated.
#
# We can create a class called BlockedDoor that inherits from the base class Door.
#
# We just override the parent class's functions of open_door() and close_door() with our own BlockedDoor version of those functions, which simply raises an Error that a blocked door cannot be opened or closed.

class Door:
    def __init__(self, is_opened=False):
        self.is_opened = is_opened

    def open_door(self):
        if not self.is_opened:
            self.is_opened = True
            print("Door is now opened.")
        else:
            print("Door is already opened.")

    def close_door(self):
        if self.is_opened:
            self.is_opened = False
            print("Door is now closed.")
        else:
            print("Door is already closed.")


class BlockedDoor(Door):
    def __init__(self, is_opened=False):
        super().__init__(is_opened)

    def open_door(self):
        raise Exception("Blocked door cannot be opened.")

    def close_door(self):
        raise Exception("Blocked door cannot be closed.")


# Example usage:
try:
    blocked_door = BlockedDoor()
    blocked_door.open_door()
except Exception as e:
    print(e)

try:
    blocked_door.close_door()
except Exception as e:
    print(e)
