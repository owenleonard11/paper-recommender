# Paper Recommender

Paper Recommender is a python package for scraping the
[University of Pennsylvania's Call for Papers site](https://call-for-papers.sas.upenn.edu/)
and recommending papers based on a title, abstract, or set of
keywords.  

## Usage

Basic scraping and viewing tasks:
```python
import papers

i = papers.Calls.CallInstance() # scrapes the site
c = i.calls[0]                  # contains the attributes of one call
print(c.description)            # prints the call description
```

Recommendation:
```

```

## Contributing

Contributions are welcome, please open an issue for major changes.

## License

[MIT](https://choosealicense.com/licenses/mit/)
