# 🌟 Exercise 1: Cats
# Key Python Topics:
#
# Classes and objects
# Object instantiation
# Attributes
# Functions
#
#
# Instructions:
#
# Use the provided Cat class to create three cat objects. Then, create a function to find the oldest cat and print its details.
#
#
#
# Step 1: Create Cat Objects
#
# Use the Cat class to create three cat objects with different names and ages.
#
#
# Step 2: Create a Function to Find the Oldest Cat
#
# Create a function that takes the three cat objects as input.
# Inside the function, compare the ages of the cats to find the oldest one.
# Return the oldest cat object.
#
#
# Step 3: Print the Oldest Cat’s Details
#
# Call the function to get the oldest cat.
# Print a formatted string: “The oldest cat is <cat_name>, and is <cat_age> years old.”
# Replace <cat_name> and <cat_age> with the oldest cat’s name and age.

class Cat:
    def __init__(self, name, age):
        self.name = name
        self.age = age


# Step 1: Create Cat Objects
cat1 = Cat("Whiskers", 5)
cat2 = Cat("Mittens", 7)
cat3 = Cat("Shadow", 3)


# Step 2: Create a Function to Find the Oldest Cat
def find_oldest_cat(*cats):
    oldest_cat = cats[0]
    for cat in cats:
        if cat.age > oldest_cat.age:
            oldest_cat = cat
    return oldest_cat


# Step 3: Print the Oldest Cat’s Details
oldest_cat = find_oldest_cat(cat1, cat2, cat3)
print(f"The oldest cat is {oldest_cat.name}, and is {oldest_cat.age} years old.")


# 🌟 Exercise 2 : Dogs
# Goal: Create a Dog class, instantiate objects, call methods, and compare dog sizes.
#
#
#
# Key Python Topics:
#
# Classes and objects
# Object instantiation
# Methods
# Attributes
# Conditional statements (if)
#
#
# Instructions:
#
# Create a Dog class with methods for barking and jumping. Instantiate dog objects, call their methods, and compare their sizes.
#
#
#
# Step 1: Create the Dog Class
#
# Create a class called Dog.
# In the __init__ method, take name and height as parameters and create corresponding attributes.
# Create a bark() method that prints “ goes woof!”.
# Create a jump() method that prints “ jumps cm high!”, where x is height * 2.
#
#
# Step 2: Create Dog Objects
#
# Create davids_dog and sarahs_dog objects with their respective names and heights.
#
#
# Step 3: Print Dog Details and Call Methods
#
# Print the name and height of each dog.
# Call the bark() and jump() methods for each dog.
#
#
# Step 4: Compare Dog Sizes

class Dog:
    def __init__(self, name, height):
        self.name = name
        self.height = height

    def bark(self):
        print(f"{self.name} goes woof!")

    def jump(self):
        print(f"{self.name} jumps {self.height * 2} cm high!")


# Step 2: Create Dog Objects
davids_dog = Dog("David's Dog", 50)
sarahs_dog = Dog("Sarah's Dog", 60)

# Step 3: Print Dog Details and Call Methods
print(f"{davids_dog.name} is {davids_dog.height} cm tall.")
davids_dog.bark()
davids_dog.jump()

print(f"{sarahs_dog.name} is {sarahs_dog.height} cm tall.")
sarahs_dog.bark()
sarahs_dog.jump()

# Step 4: Compare Dog Sizes
if davids_dog.height > sarahs_dog.height:
    print(f"{davids_dog.name} is taller than {sarahs_dog.name}.")
elif davids_dog.height < sarahs_dog.height:
    print(f"{sarahs_dog.name} is taller than {davids_dog.name}.")
else:
    print(f"{davids_dog.name} and {sarahs_dog.name} are the same height.")


# 🌟 Exercise 3 : Who’s the song producer?
# Goal: Create a Song class to represent song lyrics and print them.
#
#
#
# Key Python Topics:
#
# Classes and objects
# Object instantiation
# Methods
# Lists
#
#
# Instructions:
#
# Create a Song class with a method to print song lyrics line by line.
#
#
#
# Step 1: Create the Song Class
#
# Create a class called Song.
# In the __init__ method, take lyrics (a list) as a parameter and create a corresponding attribute.
# Create a sing_me_a_song() method that prints each element of the lyrics list on a new line.
#
#
# Example:
#
# stairway = Song(["There’s a lady who's sure", "all that glitters is gold", "and she’s buying a stairway to heaven"]

class Song:
    def __init__(self, lyrics):
        """Initialize the Song with lyrics."""
        self.lyrics = lyrics

    def sing_me_a_song(self):
        for line in self.lyrics:
            print(line)


# Step 2: Create a Song Object
stairway = Song([
    "There’s a lady who's sure",
    "all that glitters is gold",
    "and she’s buying a stairway to heaven"
])

# Step 3: Call the sing_me_a_song() Method
stairway.sing_me_a_song()


# 🌟 Exercise 4 : Afternoon at the Zoo
# Goal:
#
# Create a Zoo class to manage animals. The class should allow adding animals, displaying them, selling them, and organizing them into alphabetical groups.
#
#
#
# Key Python Topics:
#
# Classes and objects
# Object instantiation
# Methods
# Lists
# Dictionaries (for grouping)
# String manipulation
#
#
# Instructions
# Step 1: Define the Zoo Class
# 1. Create a class called Zoo.
#
# 2. Implement the __init__() method:
#
# It takes a string parameter zoo_name, representing the name of the zoo.
# Initialize an empty list called animals to keep track of animal names.
# 3. Add a method add_animal(new_animal):
#
# This method adds a new animal to the animals list.
# Do not add the animal if it is already in the list.
# 4. Add a method get_animals():
#
# This method prints all animals currently in the zoo.
# 5. Add a method sell_animal(animal_sold):
#
# This method checks if a specified animal exists on the animals list and if so, remove from it.
# 6. Add a method sort_animals():
#
# This method sorts the animals alphabetically.
# It also groups them by the first letter of their name.
# The result should be a dictionary where:
# Each key is a letter.
# Each value is a list of animals that start with that letter.
# Example output:
#
# {
#    'B': ['Baboon', 'Bear'],
#    'C': ['Cat', 'Cougar'],
#    'G': ['Giraffe'],
#    'L': ['Lion'],
#    'Z': ['Zebra']
# }
# 7. Add a method get_groups():
#
# This method prints the grouped animals as created by sort_animals().
# Example output:
#
# B: ['Baboon', 'Bear']
# C: ['Cat', 'Cougar']
# G: ['Giraffe']
# ...
#
#
# Step 2: Create a Zoo Object
# Create an instance of the Zoo class and pass a name for the zoo.
#
#
# Step 3: Call the Zoo Methods
# Use the methods of your Zoo object to test adding, selling, displaying, sorting, and grouping animals.
#
#
# Example (No Internal Logic Provided)
# class Zoo:
#     def __init__(self, zoo_name):
#         pass
#
#     def add_animal(self, new_animal):
#         pass
#
#     def get_animals(self):
#         pass
#
#     def sell_animal(self, animal_sold):
#         pass
#
#     def sort_animals(self):
#         pass
#
#     def get_groups(self):
#         pass
#
# # Step 2: Create a Zoo instance
# ramat_gan_safari = Zoo("Ramat Gan Safari")
#
# # Step 3: Use the Zoo methods
# ramat_gan_safari.add_animal("Giraffe")
# ramat_gan_safari.add_animal("Bear")
# ramat_gan_safari.add_animal("Baboon")
# ramat_gan_safari.get_animals()
# ramat_gan_safari.sell_animal("Bear")
# ramat_gan_safari.get_animals()
# ramat_gan_safari.sort_animals()
# ramat_gan_safari.get_groups()

class Zoo:
    def __init__(self, zoo_name):
        """Initialize the Zoo with a name and an empty list of animals."""
        self.zoo_name = zoo_name
        self.animals = []

    def add_animal(self, new_animal):
        """Add a new animal to the zoo if it is not already present."""
        if new_animal not in self.animals:
            self.animals.append(new_animal)

    def get_animals(self):
        """Print all animals currently in the zoo."""
        print("Animals in the zoo:")
        for animal in self.animals:
            print(animal)

    def sell_animal(self, animal_sold):
        """Remove an animal from the zoo if it exists."""
        if animal_sold in self.animals:
            self.animals.remove(animal_sold)

    def sort_animals(self):
        """Sort animals alphabetically and group them by their first letter."""
        grouped_animals = {}

        for animal in sorted(self.animals):
            first_letter = animal[0].upper()  # Ensure the first letter is uppercase for grouping

            if first_letter not in grouped_animals:
                grouped_animals[first_letter] = []  # Initialize a new list for this letter
            grouped_animals[first_letter].append(animal)
        return grouped_animals

    def get_groups(self):
        """Print the grouped animals."""
        groups = self.sort_animals()
        for letter, animals in groups.items():
            print(f"{letter}: {animals}")


# Step 2: Create a Zoo instance
casela = Zoo("Casela Zoo")
# Step 3: Use the Zoo methods
casela.add_animal("Giraffe")
casela.add_animal("Bear")
casela.add_animal("Baboon")
casela.get_animals()

casela.sell_animal("Bear")
casela.get_animals()
casela.sort_animals()
casela.get_groups()
