from bs4 import BeautifulSoup


def parse_html(content):
    soup = BeautifulSoup(content, 'lxml')
    result = ""
    for div in soup.find_all('div'):
        for p in div.find_all('p'):
            result += p.get_text()
    return result


def get_text_from_html(path: str):
    with open("files/" + path, "r") as f:
        contents = f.read()
    return parse_html(contents)


if __name__ == "__main__":
    print(get_text_from_html("SpGovEn.html"))
