from bs4 import BeautifulSoup
import urllib3
import time
import networkx as nx


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


def crawl(pool: urllib3.PoolManager, url, deep=1, sleep_time=0.5, n=5, prefix="https://en.wikipedia.org"):
    time.sleep(sleep_time)
    site = pool.request("GET", url)
    soup = BeautifulSoup(site.data, parser="lxml")
    if deep > 0:
        return (
            url,
            [crawl(pool, url_, deep - 1) for url_ in get_links_from_wiki(
                soup=soup, n=n, prefix=prefix)],
        )
    return url, get_links_from_wiki(soup=soup, n=n, prefix=prefix)


def get_edges(item):
    edges = []
    url, elements = item
    if isinstance(elements, list):
        for element in elements:
            if isinstance(element, str):
                edges.append((url.split("/")[-1], element.split("/")[-1]))
            else:
                edges.append((url.split("/")[-1], element[0].split("/")[-1]))
                edges += get_edges(element)
    return edges


def get_nodes(edges):
    nodes = []
    for edge in edges:
        nodes += list(edge)
    return nodes


def get_graph(pool,url, deep=1, sleep_time=0.5, n=5,
              prefix="https://en.wikipedia.org" ):
    tree = crawl(pool=pool, url=url, deep=deep, sleep_time=sleep_time, n=n,
                 prefix=prefix)
    edges = get_edges(tree)
    nodes = get_nodes(edges)

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph
