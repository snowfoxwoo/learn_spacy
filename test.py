import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = "SpaCy is an amazing library for natural language processing."

# Process the text with spaCy
doc = nlp(text)

# Print tokens
for token in doc:
    print(token.text)