# Path to the file
file_path = "files/nameslist.txt"

# 1. Read the file line by line
with open(file_path, "r") as f:
    lines = f.readlines()

# 2. Read only the 5th line of the file
fifth_line = lines[4].strip()

# 3. Read only the 5 first characters of the file
with open(file_path, "r") as f:
    first_five_chars = f.read(5)

# 4. Read all the file and return it as a list of strings. Then split each word
#    (since each line is a word, it's already split)
words_list = [line.strip() for line in lines]

# 5. Find out how many occurrences of the names "Darth", "Luke" and "Lea" are in the file
from collections import Counter

name_counts = Counter(words_list) # use Counter to count occurrences (performance efficient)
darth_count = name_counts["Darth"]
luke_count = name_counts["Luke"]
lea_count = name_counts["Lea"]

# 6. Append your first name at the end of the file
with open(file_path, "a") as f:
    f.write("\nGuillaume")

# 7. Append "SkyWalker" next to each first name "Luke"
with open(file_path, "r") as f:
    content = f.readlines()

with open(file_path, "w") as f:
    for line in content:
        name = line.strip()
        if name == "Luke":
            f.write("Luke SkyWalker\n")
        else:
            f.write(line)

# Output for debugging/validation
print("Fifth line:", fifth_line)
print("First 5 chars:", first_five_chars)
print("Words list:", words_list)
print(f"Counts - Darth: {darth_count}, Luke: {luke_count}, Lea: {lea_count}")
