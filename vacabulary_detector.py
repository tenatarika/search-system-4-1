import json
import requests

from alphabet_detector import AlphabetDetector
from html_reader import get_text_from_html
from nltk.probability import FreqDist

ad = AlphabetDetector()

token = ""

headers = {"Authorization": f"Bearer {token}"}
url = "https://api.edenai.run/v2/translation/language_detection"

payload = {"providers": "amazon", 'text': "this is a test"}

response = requests.post(url, json=payload, headers=headers)
result = json.loads(response.text)


class NewLangDetector:

    def frequency_detect(self, text: str):
        return "RU" if any({ad.is_latin(word) for word, _ in FreqDist(text).most_common(5)}) else "EN"

    def get_alhabet_getect(self, text: str):
        ad = AlphabetDetector()
        return "EN" if ad.is_cyrillic(text) else "RU"

    def neural_network_detect(self, text):

        token = ""
        headers = {"Authorization": f"Bearer {token}"}
        url = "https://api.edenai.run/v2/translation/language_detection"
        payload = {"providers": "amazon", 'text': f"тест"}
        response = requests.post(url, json=payload, headers=headers)
        return json.loads(response.text)

    def get_language(self, file_path: str):
        text = get_text_from_html(file_path)
        return {
            "frequency_detect": self.frequency_detect(text),
            "get_alhabet_getect": self.get_alhabet_getect(text),
            "neural_network_detect": self.neural_network_detect(text)
        }


if __name__ == "__main__":
    h = NewLangDetector()
    print(h.get_language("rutext.html"))

    print(ad.only_alphabet_chars(u"ελληνικά means greek", "LATIN"))
    eng_text = get_text_from_html("SpGovEn.html")
    ru_text = get_text_from_html("rutext.html")

    fdist = FreqDist(eng_text)

    # fdist.plot(30, cumulative=False)
    print(fdist.most_common(5))
    for word, total in fdist.most_common(5):
        print(ad.is_latin(word))

    # print(detect("здесь произвольный фрагмент текста на вход"))
    print(ad.is_cyrillic(ru_text[:10]))

    print(result['amazon']['items'])
