"""
Anagram Checker Module

This module provides the AnagramChecker class which handles all anagram-related logic
including word validation and anagram finding functionality.

Classes:
    AnagramChecker: A class for checking word validity and finding anagrams

Dependencies:
    - sowpods.txt: A text file containing English words (one per line)
"""


class AnagramChecker:
    """
    A class for checking if words are valid English words and finding their anagrams.

    This class loads a dictionary of English words from a text file and provides
    methods to validate words and find anagrams.

    Attributes:
        word_list (set): A set containing all valid English words in lowercase
    """

    def __init__(self):
        """
        Initialize the AnagramChecker by loading the word list from file.

        Loads all words from 'sowpods.txt' into a set for efficient lookup.
        Words are converted to lowercase and stored without whitespace.

        Raises:
            FileNotFoundError: If sowpods.txt is not found, an error message is printed
                              and an empty word list is created.
        """
        self.word_list = set()
        try:
            with open('sowpods.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    word = line.strip().lower()
                    if word:  # Only add non-empty words
                        self.word_list.add(word)
        except FileNotFoundError:
            print("Error: sowpods.txt file not found!")
            self.word_list = set()

    def is_valid_word(self, word):
        """
        Check if the given word is a valid English word.

        Args:
            word (str): The word to validate (case-insensitive)

        Returns:
            bool: True if the word exists in the dictionary, False otherwise
        """
        return word.lower() in self.word_list

    @staticmethod
    def is_anagram(word1, word2):
        """
        Check if two words are anagrams of each other.

        Two words are anagrams if they contain exactly the same letters
        but in a different order. The words must be different from each other.

        Args:
            word1 (str): First word to compare (case-insensitive)
            word2 (str): Second word to compare (case-insensitive)

        Returns:
            bool: True if the words are anagrams, False otherwise
        """
        # Convert to lowercase and sort the letters
        return sorted(word1.lower()) == sorted(word2.lower()) and word1.lower() != word2.lower()

    def get_anagrams(self, word):
        """
        Find all anagrams for the given word from the dictionary.

        Searches through the entire word list to find words that are anagrams
        of the input word. Returns a sorted list of anagrams.

        Args:
            word (str): The word to find anagrams for (case-insensitive)

        Returns:
            list: A sorted list of anagram strings in lowercase.
                 Returns an empty list if no anagrams are found.
        """
        anagrams = []
        word_lower = word.lower()

        # Check each word in our word list
        for dictionary_word in self.word_list:
            if self.is_anagram(word_lower, dictionary_word):
                anagrams.append(dictionary_word)

        return sorted(anagrams)
