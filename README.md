# Paper Recommender

Paper Recommender is a python package for scraping the
[University of Pennsylvania's Call for Papers site](https://call-for-papers.sas.upenn.edu/)
and recommending papers based on a title, abstract, or set of
keywords.  

## Usage

To get started, we define a `CallInstance` Object:
```python
import papers

instance = papers.Calls.CallInstance()
```
When `instance` is initialized, the site is scraped using `requests`, `lxml`, and `BeautifulSoup` and the relevant data stored.  This may take several minutes, since each call's site must be accessed individually for a full description and there is some wait time between requests.  Although the `instance` object is mostly used for the various recommender functions, its individual `Call` Objects may be accessed using the `calls` attribute:
```python
call = instance[index] # A single Call Object retrieved from instance
call[source]           # The title of the Call
```
To get a set of recommendations, we use either `keyword_recommend`, which takes a list of keywords, `abstract_recommend`, which takes a longer body of text, or `title_recommend`:
```python
recs = instance.title_recommend('Some Paper Title')
```
Returned from all of these methods is a `RecList` Object, which is really just a list of `CallRec` Objects.  The important method is `show`, which takes no parameters and returns a markdown representation of the recommendation set.  For example, to view the nicely formatted recommendations we retrieved above we would use:
```python
from IPython import display

display(recs.show())
```

## Note for Technical Users
It is also possible to ignore the recommendation features of the package and use it as a convenient web scraper if further information about or analysis of the site is desired.  Most of the important attributes of a listing are contained in the `Call` Object; more details can be found in its docstring.

## Contributing

Contributions are welcome, please open an issue for major changes.

## License

[MIT](https://choosealicense.com/licenses/mit/)
