from bs4 import BeautifulSoup
import urllib3
import time


def get_links_from_wiki(soup, n=5, prefix="https://en.wikipedia.org"):
    arr = []
    for i, a in enumerate(
        soup.find("div", class_="mw-parser-output").find("p").find_all("a", href=True)
    ):
        if len(arr) > n:
            break
        if a["href"].startswith("/wiki"):
            arr.append(prefix + a["href"])
    return arr


def crawl(pool: urllib3.PoolManager, url, deep=1, sleep_time=0.5):
    time.sleep(sleep_time)
    site = pool.request("GET", url)
    soup = BeautifulSoup(site.data, parser="lxml")
    if deep > 0:
        return (
            url,
            [crawl(pool, url_, deep - 1) for url_ in get_links_from_wiki(soup=soup)],
        )
    return (url, get_links_from_wiki(soup=soup))


def edges(item):
    edges = []
    url, elements = item
    if isinstance(elements, list):
        for element in elements:
            if isinstance(element, str):
                edges.append((url, element))
            else:
                edges.append((url, element[0]))
                edges += edges(element)
    return edges
