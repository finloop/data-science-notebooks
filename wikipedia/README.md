```python
from bs4 import BeautifulSoup
import urllib3
import time
```

# How fast can we get to [philosophy](https://en.wikipedia.org/wiki/Philosophy) ?

## Hypothesis
In this experiment, I'll test the hypothesis that:
**By going to the first link on any Wikipedia article, you'll end up on the [philosophy](https://en.wikipedia.org/wiki/Philosophy) article.** 

## Solution
To do this, I simplified the problem to two smaller problems:
- Getting links from article (parsing article)
- Downloading article and building URL tree

For each article I'll enter the first URL, if that URL contains the phrase`Philosophy` the algorithm will end.


`get_links_from_wiki` function parses the article. It works by finding a div that contains the whole article, then iterates through all paragraphs and finds all links that match pattern `/wiki/article_name`. Because there's no domain in that pattern, it is added at the end.


```python
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
        # In each paragraph find all <a href="/wiki/article_name"></a> and extract "/wiki/article_name"
        for i, a in enumerate(
            paragraph.find_all("a", href=True)
        ):
            if len(arr) >= n:
                break
            if a["href"].startswith("/wiki") and len(a["href"].split("/")) == 3 :
                arr.append(prefix + a["href"])
    return arr
```

The crawl function will be recursive, for each URL found on page I'll call it again. For rach iteration it'll check if URL contains that phrase, if so it'll return both the site and link to Philosophy. To control number of recursive calls, depth of created tree is limited by `depth` parameter. 


```python
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
```

## The experiment


```python
# Instance of PoolManager that each crawler will share
pool = urllib3.PoolManager() 
```

To test the hypothesis we'll start from page `https://en.wikipedia.org/wiki/Data_mining"`, look for page `Philosophy` and set link limit for crawler to `1` so that it'll only enter the first link on each page.


```python
crawl(pool, "https://en.wikipedia.org/wiki/Data_mining", phrase="Philosophy", deep=30, n=1, verbose=True)
```

    30 Entering Data_mining
    29 Entering Data_set
    28 Entering Data
    27 Entering American_English
    26 Entering Variety_(linguistics)
    25 Entering Sociolinguistics
    24 Entering Society
    23 Entering Social_group
    22 Entering Social_science
    21 Entering Branches_of_science
    20 Entering Science
    19 Entering Latin_language
    18 Entering Classical_language
    17 Entering Language
    16 Entering Communication
    15 Entering Academic_discipline
    14 Entering Knowledge
    13 Entering Fact
    12 Entering Experience
    11 Entering Consciousness
    10 Entering Sentience
    9 Entering Emotion
    8 Entering Mental_state
    7 Entering Mind
    6 Entering Thought
    5 Entering Ideas





    ('https://en.wikipedia.org/wiki/Data_mining',
     [('https://en.wikipedia.org/wiki/Data_set',
       [('https://en.wikipedia.org/wiki/Data',
         [('https://en.wikipedia.org/wiki/American_English',
           [('https://en.wikipedia.org/wiki/Variety_(linguistics)',
             [('https://en.wikipedia.org/wiki/Sociolinguistics',
               [('https://en.wikipedia.org/wiki/Society',
                 [('https://en.wikipedia.org/wiki/Social_group',
                   [('https://en.wikipedia.org/wiki/Social_science',
                     [('https://en.wikipedia.org/wiki/Branches_of_science',
                       [('https://en.wikipedia.org/wiki/Science',
                         [('https://en.wikipedia.org/wiki/Latin_language',
                           [('https://en.wikipedia.org/wiki/Classical_language',
                             [('https://en.wikipedia.org/wiki/Language',
                               [('https://en.wikipedia.org/wiki/Communication',
                                 [('https://en.wikipedia.org/wiki/Academic_discipline',
                                   [('https://en.wikipedia.org/wiki/Knowledge',
                                     [('https://en.wikipedia.org/wiki/Fact',
                                       [('https://en.wikipedia.org/wiki/Experience',
                                         [('https://en.wikipedia.org/wiki/Consciousness',
                                           [('https://en.wikipedia.org/wiki/Sentience',
                                             [('https://en.wikipedia.org/wiki/Emotion',
                                               [('https://en.wikipedia.org/wiki/Mental_state',
                                                 [('https://en.wikipedia.org/wiki/Mind',
                                                   [('https://en.wikipedia.org/wiki/Thought',
                                                     [('https://en.wikipedia.org/wiki/Ideas',
                                                       ['https://en.wikipedia.org/wiki/Philosophy'])])])])])])])])])])])])])])])])])])])])])])])])])])



As you can see after 25 iterations indeed we found `Philosophy` page.
