```python
import spacy

# Load the English model
nlp = spacy.load("en_core_web_sm")

# Sample text
text = """
Apple is looking at buying U.K. startup for $1 billion. 
The quick brown foxes were jumping over the lazy dogs.
"""

# Process the text
doc = nlp(text)

# Tokenization and Lemmatization
for token in doc:
    print(f"{token.text} {token.lemma_}")
```