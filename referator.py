from rake_nltk import Rake

r = Rake()
text="Feature extraction is not that complex. There are many algorithms available that can help you with feature extraction. Rapid Automatic Key Word Extraction is one of those"
r.extract_keywords_from_text(text)




if __name__ == '__main__':
    print(r.get_ranked_phrases_with_scores())
    print(r.get_ranked_phrases())
