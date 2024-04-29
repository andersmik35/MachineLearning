from spellchecker import SpellChecker

def spell_check_text(text):
    # Create a spell checker instance
    spell = SpellChecker()

    # Find the words in the text
    words = text.split()

    # Find those words that may be misspelled
    misspelled = spell.unknown(words)

    corrections = {}
    for word in misspelled:
        # Get the one `most likely` answer
        correct = spell.correction(word)
        # Get a list of `likely` options
        candidates = spell.candidates(word)

        corrections[word] = {
            'correction': correct,
            'candidates': candidates
        }

    return corrections

# Example usage
text = "speling erors in somethink as simple as writting a sentance."
corrections = spell_check_text(text)

print("Corrections Needed:")
for misspelled, details in corrections.items():
    print(f"{misspelled} -> {details['correction']} (Candidates: {details['candidates']})")
