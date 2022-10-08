import os
import itertools
import math
from typing import List, Union, Any, Optional, Dict
from numpy import dot
from numpy.linalg import norm
from pydantic import BaseModel, Field


class File(BaseModel):
    vector_word: Dict[str, int]
    filename: str
    TF: Optional[Any]
    hit: Optional[float]


class SearchEngine:

    def __init__(self):
        self.all_TF = []
        self.files = []

    def get_unic_words_from_file(self, filename):
        with open(filename, 'r') as file:
            text = file.read()
            text = text.lower()
            words = text.split()
            words = [word.strip('.,!;()[]') for word in words]  # cleaning
            words = [word.replace("'s", '') for word in words]
            unique = []
            for word in words:
                if word not in unique:
                    unique.append(word)
            return unique

    def get_unic_words_for_each_file(self):
        files_with_unic_words = []
        for filename in os.listdir("files"):
            unic_words = self.get_unic_words_from_file(os.path.join("files", filename))
            files_with_unic_words.append(unic_words)
        return files_with_unic_words

    def get_row_text(self, filename):
        with open(os.path.join("files", filename), 'r') as f:
            text = f.read()
        return text

    def get_tf(self):
        file_unic_words = self.get_unic_words_for_each_file()
        res = []
        for structure in file_unic_words:
            for filename, unic_words in structure.items():
                row_text = self.get_row_text(filename)
                file = []
                for u_word in unic_words:
                    count_unic_word = row_text.count(u_word)
                    file.append({u_word: count_unic_word})
                res.append({filename: file})
        return res

    def new_unic_words(self):
        files_with_unic_words = []
        for filename in os.listdir("files"):
            unic_words = self.get_unic_words_from_file(os.path.join("files", filename))
            for word in unic_words:
                files_with_unic_words.append(word)
        return files_with_unic_words

    def create_matrix_curr(self) -> List[File]:
        """list of dicts it word on in unic word"""
        unic_words = self.new_unic_words()
        num_of_words_all_files = []
        for filename in os.listdir("files"):
            with open(os.path.join("files", filename), 'r') as f:
                text = f.read()
                numOfWords = dict.fromkeys(unic_words, 0)
                for nword in text.lower().split(" "):
                    numOfWords[nword] += 1
                num_of_words_all_files.append(File(filename=filename, vector_word=numOfWords))
        return num_of_words_all_files

    def computeOneTF(self, word_count: Dict[str, int], all_words: List[str]):
        tfDict = {}
        bagOfWordsCount = len(all_words)
        for word, count in word_count.items():
            tfDict[word] = count / float(bagOfWordsCount)
        return tfDict

    def computeTF(self):
        res = []
        bag_words = self.get_unic_words_for_each_file()
        tr_bag_words = list(itertools.chain(*bag_words))
        for word_d in self.create_matrix_curr():
            ans = self.computeOneTF(word_count=word_d.vector_word, all_words=tr_bag_words)
            word_d.TF = ans
            res.append(ans)
            self.files.append(word_d)
        return res

    def computeIDF(self, documents: List[File]):
        N = len(documents)

        idfDict = dict.fromkeys(documents[0].vector_word.keys(), 0)
        for document in documents:
            for word, val in document.vector_word.items():
                if val > 0:
                    idfDict[word] += 1

        for word, val in idfDict.items():
            idfDict[word] = math.log(N / float(val))
        return idfDict

    def computeTFIDF(self, tfBagOfWords):
        idfs = self.computeIDF(self.create_matrix_curr())
        tfidf = {}
        for word, val in tfBagOfWords.items():
            tfidf[word] = val * idfs[word]
        return tfidf

    def get_files(self):
        files = []
        for filename in os.listdir("files"):
            print(filename)
            files.append(filename)
            with open(os.path.join("files", filename), 'r') as f:
                text = f.read()
                print(text)
        return files

    def create_query_matrix_curr(self, query: str):
        """list of dicts it word on in unic word"""

        unic_words = self.new_unic_words()
        numOfWords = dict.fromkeys(unic_words, 0)
        for nword in query.lower().split(" "):
            numOfWords[nword] += 1
        return numOfWords

    def calculate_res(self, query: str):
        curr_query = self.create_query_matrix_curr(query)
        results = []
        for TF in self.computeTF():
            TFIDF = list(self.computeTFIDF(TF).values())
            hit = dot(list(curr_query.values()), TFIDF) / (norm(list(curr_query.values())) * norm(TFIDF))
            results.append(hit)
            for file in self.files:
                if file.TF == TF:
                    file.hit = hit

        return results
