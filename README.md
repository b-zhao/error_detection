# possible models
1. Design features -> design similarity function -> generative model (input: affinity matrix, output: labels)
2. Design features -> generative model (input: features, output: labels)

Labels are {erroneous, not erroneous}

Possible choices of features: numerical columns can remain the same; categorical columns can be converted to one-hot; text can be encoded using word2vec
