import spacy

nlp = spacy.load("en_core_web_md")

doc2 = nlp("John McCain and I visited the Apple Store in Manhattan.")

for item in doc2.ents:
    print(item)

for item in doc2.ents:
    print (item.text, item.label_)


doc3 = nlp(open("EightThings.txt").read())


for item in doc3.ents:
    print (item.text, item.label_)





