# Extractors


## Level
- Word level: each word has a feature value, like TF-IDF, or word index
- Sentence level: each sentence has a value, like sentence length
- Pair level: each pair has a value, like TF-IDF similarity between the two sentences


## Type
- Embed: the feature values are integers, and will be converted into a vector by a embedding matrix
- Float: the feature values are float numbers
- Other: like label or sid, not used as features


## Extractor List

### 1. WordEmbedExtractor
1. s1/2_word: word level, embed type
2. s1/2_len: sentence level, float type (int data type)
3. label: pair level, other type
4. sid: pair level, other type


### 2. TFIDFExtractor
