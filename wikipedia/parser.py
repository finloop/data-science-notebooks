from bs4 import BeautifulSoup
import urllib3
import time
import networkx as nx


def get_links_from_wiki(soup, n=5, prefix="https://en.wikipedia.org"):
    """
    Extracts `n` first links from wikipedia articles and adds `prefix` to
    internal links.

    Parameters
    ----------
    soup : BeautifulSoup
        Wikipedia page
    n : int
        Number of links to return
    prefix : str, default="https://en.wikipedia.org""
        Site prefix
    Returns
    -------
    list
        List of links
    """
    arr = []

    # Get all paragraphs
    for paragraph in soup.find("div", class_="mw-parser-output").find_all("p"):
        # In each paragraph find all <a href="/wiki/article_name"></a> and
        # extract "/wiki/article_name"
        for i, a in enumerate(
                paragraph.find_all("a", href=True)
        ):
            if len(arr) >= n:
                break
            if a["href"].startswith("/wiki") and len(a["href"].split("/")) == 3:
                arr.append(prefix + a["href"])
    return arr


def crawl(
    pool: urllib3.PoolManager,
    url,
    phrase=None,
    deep=1,
    sleep_time=0.5,
    n=5,
    prefix="https://en.wikipedia.org",
    verbose=False,
):
    """
    Crawls given Wikipedia `url` (article) with max depth `deep`. For each page
    extracts `n` urls and  if `phrase` is given check if `phrase` in urls.

    Parameters
    ----------
    pool : urllib3.PoolManager
        Request pool
    phrase : str
        Phrase to search for in urls.
    url : str
        Link to wikipedia article
    deep : int
        Depth of crawl
    sleep_time : float
        Sleep time between requests.
    n : int
        Number of links to return
    prefix : str, default="https://en.wikipedia.org""
        Site prefix

    Returns
    -------
    tuple
        Tuple of url, list
    """
    if verbose:
        site = url.split("/")[-1]
        print(f"{deep} Entering {site}")

    time.sleep(sleep_time)
    site = pool.request("GET", url)
    soup = BeautifulSoup(site.data, parser="lxml")
    links = get_links_from_wiki(soup=soup, n=n, prefix=prefix)
    is_phrase_present = any([phrase in link for link in links]) and phrase is not None
    if deep > 0 and not is_phrase_present:
        return (
            url,
            [
                crawl(
                    pool=pool,
                    url=url_,
                    phrase=phrase,
                    deep=deep - 1,
                    sleep_time=sleep_time,
                    n=n,
                    prefix=prefix,
                    verbose=verbose,
                )
                for url_ in links
            ],
        )
    return url, links


def get_edges(tree):
    """
    Get edges from url tree. Where url tree is tuple of (url, list of tuples(
    url, list)) etc.

    Example tree:
    (url, [(url1, [...]),
          (url2, [...]).
    ])

    Parameters
    ----------
    tree : tuple
        Tree of urls.

    Returns
    -------
    list
        List of tuples (source page, end page).
    """
    edges = []
    url, elements = tree
    if isinstance(elements, list):
        for element in elements:
            if isinstance(element, str):
                edges.append((url.split("/")[-1], element.split("/")[-1]))
            else:
                edges.append((url.split("/")[-1], element[0].split("/")[-1]))
                edges += get_edges(element)
    return edges


def get_nodes(edges):
    """
    Get nodes from list of edges.

    Parameters
    ----------
    edges : list
         List of tuples (source page, end page).
    Returns
    -------
    list
        nodes, not unique
    """
    nodes = []
    for edge in edges:
        nodes += list(edge)
    return nodes


def get_graph(
    pool, url, deep=1, sleep_time=0.5, n=5, prefix="https://en.wikipedia.org"
):
    """
    Generates link graph for given Wikipedia article.

    Parameters
    ----------
    pool : urllib3.PoolManager
        Request pool
    url : str
        Link to wikipedia article
    deep : int
        Depth of crawl
    sleep_time : float
        Sleep time between requests.
    n : int
        Number of links to return
    prefix : str, default="https://en.wikipedia.org""
        Site prefix
    Returns
    -------
    nx.DiGraph
        Graph of url.
    """
    tree = crawl(
        pool=pool, url=url, deep=deep, sleep_time=sleep_time, n=n, prefix=prefix
    )
    edges = get_edges(tree)
    nodes = get_nodes(edges)

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph
