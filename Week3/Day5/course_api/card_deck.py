class Card:
    SUITS = ['Hearts', 'Diamonds', 'Clubs', 'Spades']
    VALUES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']

    def __init__(self, suit, value):
        self.suit = suit  # triggers setter/validation
        self.value = value

    def __str__(self):
        """
        Converts the object to its string representation.

        This method is meant to provide a human-readable string representing
        the object's state, useful for logging or debugging purposes.

        :return: Human-readable string representation of the object.
        :rtype: str
        """
        return f"{self.value} of {self.suit}"

    def __repr__(self):
        """
        Provides a string representation of the object. This representation is intended to
        be unambiguous, showing essential details about the object to developers, typically
        used for debugging purposes.

        :return: A string representation of the object detailing its ``suit`` and ``value``
                 attributes.
        :rtype: str
        """
        return f"Card(suit='{self.suit}', value='{self.value}')"

    @property
    def suit(self):
        return self._suit

    @suit.setter
    def suit(self, suit):
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit!r}. Allowed: {self.SUITS}")
        self._suit = suit

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if value not in self.VALUES:
            raise ValueError(f"Invalid value: {value!r}. Allowed: {self.VALUES}")
        self._value = value


class Deck:
    def __init__(self):
        self._cards = [
            Card(suit, value)
            for suit in Card.SUITS
            for value in Card.VALUES
        ]
        self._validate_deck()

    @property
    def cards(self):
        return list(self._cards)

    def _validate_deck(self):
        if len(self._cards) != 52:
            raise ValueError(f"Deck must have 52 cards, found {len(self._cards)}.")

    def __repr__(self):
        return f"Deck({len(self._cards)} cards)"


# Demo usage
if __name__ == "__main__":
    deck = Deck()
    print(deck)  # Deck(52 cards)
    print(deck.cards[:5])  # Show first 5 cards
    print(deck.cards[0])  # A of Hearts
    # Test validation
    try:
        bad_card = Card('Stars', 'A')
    except ValueError as e:
        print(f"Validation works: {e}")
