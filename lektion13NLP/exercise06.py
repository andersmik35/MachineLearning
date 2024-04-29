import spacy

# Load the English NER model
nlp = spacy.load("en_core_web_md")

# Example text
text = "Apple is looking at buying U.K. startup for $1 billion"

# Process the text
doc = nlp(text)

# Extract entities
for ent in doc.ents:
    print(ent.text, ent.label_)
