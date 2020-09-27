import datetime
from time import sleep

import sklearn.feature_extraction.text as sci_text
from scipy.spatial.distance import cdist
from IPython.display import Markdown

from bs4 import BeautifulSoup
import requests
import lxml.html

import nltk
nltk.download("wordnet")


# Used in the Call.parse_due_date function to associate a month name with the corresponding integer
MONTH_DICT = dict(January=1, February=2, March=3, April=4, May=5, June=6, July=7, August=8, September=9, October=10,
                  November=11, December=12)

# Setting custom stopwords for nltk processes
NLTK_STOPWORDS = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself",
                  "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself",
                  "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
                  "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
                  "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as",
                  "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
                  "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off",
                  "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how",
                  "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
                  "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
                  "now"]

# Determined somewhat un-scientifically
CUSTOM_STOPWORDS = ["journal", "publish", "deadline", "open", "access", "submission", "scope", "article", "papers",
                    "please", "abstract", "submit", "word", "literature", "new", "ha", "study", "panel", "question",
                    "work", "include", "also", "proposal", "submitted", "topic", "welcome", "limited", "various",
                    "theme", "send", "address", "author", "conference"]

STOPWORDS = NLTK_STOPWORDS + CUSTOM_STOPWORDS


class Call:
    """A single call from english.upenn.edu's call for papers site.

    Holds parsed attributes of a call and provides methods for their access and manipulation.  Intended mostly for use
    within the CallInstance class.

    ...

    Attributes
    ----------
    article : BeautifulSoup
        a BeautifulSoup object containing the call's search page

    source : str
        the source of the call, often a conference or journal

    source_link : str
        link to a page containing more details about the call

    updated : str
        a string of the form "Weekday, Month DD, YYYY - HH:MM" representing the date the call was last updated

    contact : str
        the contact provided in the call

    deadline : str
        a string of the form "Weekday, Month DD, YYYY"

    description : str
        the description provided for the call

    long_desc : str
        the long description provided for the call on its individual page

    categories : list of str
        list of categories designated for the call on its individual page

    contact_email : str
        email provided for the call's contact on its individual page

    long_desc_keywords : list of str
        tokenized and lemmatized long_desc, value will be None until the Call is part of a CallInstance

    Methods
    -------
    parse_due_date : datetime.date
        returns a representation of the string attribute deadline as a datetime.date object

    compute_idf : sparse matrix
        returns a tfidf array relative to the passed transformer

    get_long_desc_keywords : list of str
        keywords from the call's long description, sets the long_desc_keywords
    """

    def __init__(self, article):
        """
        Parameters
        ----------
        article : BeautifulSoup
            article element containing details on the call
        """

        # To be set later when a CallInstance Object is initialized
        self.long_desc_keywords = None

        # Using BeautifulSoup to retrieve each aspect of the call
        self.article = article
        self.source = article.header.h2.a.get_text()
        self.source_link = "https://call-for-papers.sas.upenn.edu" + article.header.h2.a.get('href')
        self.updated = article.find(class_='field-name-field-cfp-updated').\
            find(class_='field-items').div.span.get_text()
        self.contact = article.find(class_='field-name-field-cfp-contact-name').\
            find(class_='field-items').div.get_text()
        self.deadline = article.find(class_='field-name-field-cfp-due-date').\
            find(class_='field-items').div.span.get_text()
        self.description = article.find(class_='field-name-field-cfp-content').\
            find(class_='field-items').div.get_text()

        # Using BeautifulSoup to access the call's individual page
        html = requests.get(self.source_link)
        soup = BeautifulSoup(html.text, 'lxml')

        # Getting additional attributes from the call's individual page
        self.long_desc = soup.find(class_="field-name-field-cfp-content").get_text()
        self.categories = [div.get_text() for div in soup.find(class_="field-name-field-cfp-categories")
                           .div.find_all('div')]
        self.contact_email = soup.find(class_="field-type-email").a.get('href')

    def parse_due_date(self):
        """Returns a representation of the string attribute deadline as a datetime.date object.

        To understand this function, recall that self.deadline has the form:
            "Weekday, Month DD, YYYY"
        """

        split_deadline = self.deadline.split(',')
        month_day = split_deadline[1].split(' ')

        month = MONTH_DICT[month_day[1]]
        day = int(month_day[2])
        year = int(split_deadline[2][1:])

        return datetime.date(year, month, day)

    def get_long_desc_keywords(self, tokenizer, lemmatizer):
        """Returns a list of keywords in the description, parsed using the nltk library.

        Parameters
        ----------
        tokenizer : nltk tokenizer
            must have a tokenize() method, used to split the title string

        lemmatizer : nltk lemmatizer
            must have a lemmatizer() method, used to reduce the strings to stems

        Returns
        -------
        words : list of str
            a list of lemmatized, tokenized strings filtered through STOPWORDS
        """

        words = []
        for word in tokenizer.tokenize(self.long_desc):
            stem = lemmatizer.lemmatize(word.lower())
            if stem not in words and stem not in STOPWORDS and stem.isalpha():
                words.append(stem)

        self.long_desc_keywords = words
        return words


class CallRec:
    """An object containing a call (see documentation for Call) recommendation.

    Holds a Call Object, its relevancy, and the information about criteria used to recommend it.  Mostly intended
    as a return type for the various recommend methods of the CallInstance class.  Sortable by relevancy.

    ...

    Attributes
    ----------
    call : Call
        the base of the class, representing one recommended Call Object

    relevancy : float
        how relevant the call was to the criteria

    rec_type : str
        "keyword", "title", or "abstract", the recommender function used

    criteria
        criteria used to recommend the call, see recommender functions for details

    rec_info
        information on how relevancy was determined from the search criteria, see recommender functions for details

    Methods
    -------
    show : IPython.display.Markdown
        a markdown representation of the recommendation
    """

    def __init__(self, call, relevancy, rec_type, criteria, rec_info):
        self.call = call
        self.relevancy = relevancy
        self.rec_type = rec_type
        self.criteria = criteria
        self.rec_info = rec_info


class CallInstance:
    """An instance of english.upenn.edu's call for papers site.

    Grabs text from a certain number of pages of the site and organizes the various attributes of each call into
    convenient containers for later use by a recommendation algorithm.  A list of Call objects (see above) is the
    core of each CallInstance object.

    ...

    Attributes
    ----------
    BASE_URL : str
        the site URL, missing only the page number

    calls : list of Call
        a list of Call objects, each representing a single call and its parsed attributes

    vectorizer : nltk CountVectorizer
        used to transform the lists of keywords into objects understandable for tfidf calculation

    transformer : nltk TfidfTransformer
        used to transform vectorized lists into tfidf arrays

    Methods
    -------
    keyword_recommend : list of CallRec
        recommends papers based on a list of keywords, see README for algorithm and implementation details

    title_recommend : list of CallRec
        recommends papers based on a title, see README for algorithm and implementation details

    abstract_recommend : list of CallRec
        recommends papers based on an abstract, see README for algorithm and implementation details
    """

    # URL for call-for-papers, same for all instances
    BASE_URL = "https://call-for-papers.sas.upenn.edu/category/all&page="

    def __init__(self, scope="default", n=0):
        """
        Parameters
        ----------
        scope : str
            * default : stops scraping after a full page of overdue calls
            * pages   : scrapes exactly n pages
            * calls   : scrapes exactly n calls
            * overdue : scrapes until n consecutive overdue calls are encountered

        n : int
            determines behavior of scope, see above

        Raises
        ------
        RuntimeError
            If scope expects n and none is provided
        """

        # Will be fit later
        self.transformer = sci_text.TfidfTransformer(smooth_idf=True, use_idf=True)

        # Ensures scope and helper parameters are passed appropriately
        if scope in ['pages', 'calls', 'overdue'] and not n:
            raise RuntimeError("Scope expects n to be passed to constructor but none was found.")

        # Creates the empty list self.calls
        self.calls = []

        # Adds appropriate Call objects to self.calls, accounting for scope
        page = 0
        calls = 0
        overdue_ctr = 0
        while True:

            # For the 'default' scope, keeps track of whether a non-overdue call has been found on the current page
            has_not_overdue = False

            # Performs the actual request
            url = self.BASE_URL + str(page)
            html = requests.get(url)
            soup = BeautifulSoup(html.text, 'lxml')

            for div in soup('div'):
                if 'views-row' in div.get('class', []):

                    self.calls.append(Call(div.article))
                    sleep(1)

                    # Checks whether the call is overdue, updates appropriate variables

                    if self.calls[calls].parse_due_date() < datetime.date.today():
                        overdue_ctr += 1
                    else:
                        has_not_overdue = True

                    # Call is finished, so increment calls
                    calls += 1

                # Checks whether calls or date conditions have been reached
                if scope == 'calls' and n == calls:
                    break
                if scope == 'overdue' and overdue_ctr == n:
                    break

            # Checks again to break out of outer loop
            if scope == 'calls' and n == calls:
                break
            if scope == 'overdue' and overdue_ctr == n:
                break

            # Page is finished, so increment page
            page += 1

            # Checks whether pages condition has been reached
            if scope == 'pages' and n == page:
                break

            # Checks whether default condition has been reached
            if scope == 'default' and not has_not_overdue:
                break

            # In line with the site's robots.txt
            sleep(10)

        # Preparing and fitting the TfidfTransformer for later use with recommendations
        self.vectorizer = sci_text.CountVectorizer()
        tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
        lemmatizer = nltk.WordNetLemmatizer()

        desc_lists = [' '.join(call.get_long_desc_keywords(tokenizer, lemmatizer)) for call in self.calls]
        desc_lists_vec = self.vectorizer.fit_transform(desc_lists)

        self.instance_tfidf = self.transformer.fit_transform(desc_lists_vec).toarray()

    def relevance(self, words):
        """Computes the relevance of a set of words with the entire set of calls as a reference.

        Only intended for use by the various recommender methods.

        Parameters
        ----------
        words : list of str
            the keywords to be compared against each call

        Returns
        -------
        rec_index : list of float
            base relevance (the higher the better) for
        """

        words_str = [' '.join(words)]
        words_tfidf = self.transformer.transform(self.vectorizer.transform(words_str)).toarray()
        rec_index = cdist(words_tfidf, self.instance_tfidf, 'cosine')

        return [(1 / item - 1) * 10 if item else 0 for item in rec_index[0]]

    def keyword_recommend(self, keywords, min_relevancy=0.3):
        """Recommends papers based on a list of keywords.

        The basic algorithm here is scikit-learn's tf-idf model.  The model is automatically fit to the set of calls
        when a CallInstance Object is initialized.

        ...

        Parameters
        ----------
        keywords : list of str
            a list of inputs that will be used as keywords for the recommendation algorithm

        min_relevancy : float
            the minimum relevancy required for a call to appear in the results

        Returns
        -------
        rec_list : list of CallRec
            a list of calls with relevancy greater than min_relevancy and containing additional information (see below)

        CallRec Specifications
        ----------------------
        criteria : list of str
            the list of keywords, equivalent to the keywords parameter

        rec_info : list of str
            identifies the keywords shared by the call
        """

        # A list of CallRec Objects, to be returned by the function
        rec_list = []

        # Associates each call with a relevance score
        index_zip = zip(self.calls, self.relevance(keywords))

        # Finds sufficiently relevant calls and places the resulting CallRec object in rec_list
        for call, rel in index_zip:
            if rel > min_relevancy:

                # Determines shared words for the CallRec's rec_info attribute
                shared_words = [word for word in keywords if word in call.long_desc_keywords]
                if len(shared_words) == 1:
                    ret_str = "Based on your search for keyword " + shared_words[0] + "."
                elif len(shared_words) > 1:
                    ret_str = "Based on your search for keywords " + ', '.join(shared_words[:-1]) + ' and ' + \
                              shared_words[-1] + '.'
                else:
                    ret_str = "No information about this recommendation is available."

                rec = CallRec(call, rel, "keyword", keywords, ret_str)
                rec_list.append(rec)

        return rec_list

    def abstract_recommend(self, abstract, min_relevancy=0.3):
        """Recommends papers based on an abstract.

        The basic algorithm here is scikit-learn's tf-idf model.  The model is automatically fit to the set of calls
        when a CallInstance Object is initialized.

        ...

        Parameters
        ----------
        abstract : str
            a string theoretically representing any corpus of relevant words, intended to be an abstract

        min_relevancy : float
            the minimum relevancy required for a call to appear in the results

        Returns
        -------
        rec_list : list of CallRec
            a list of calls with relevancy greater than min_relevancy and containing additional information (see below)

        CallRec Specifications
        ----------------------
        criteria : str
            a string, equivalent to the abstract parameter

        rec_info : list of str
            brief summary of the abstract used to perform the search
        """

        # A list of CallRec Objects, to be returned by the function
        rec_list = []

        # Associates each call with a relevance score
        words = []
        tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
        lemmatizer = nltk.WordNetLemmatizer()
        for word in tokenizer.tokenize(abstract):
            stem = lemmatizer.lemmatize(word.lower())
            if stem not in words and stem not in STOPWORDS and stem.isalpha():
                words.append(stem)

        index_zip = zip(self.calls, self.relevance(words))

        # Finds sufficiently relevant calls and places the resulting CallRec object in rec_list
        for call, rel in index_zip:
            if rel > min_relevancy:

                # Makes a brief summary
                ret_str = "Based on the abstract beginning \"" + " ".join(abstract.split()[:6]) + "..." + "\""

                rec = CallRec(call, rel, "abstract", words, ret_str)
                rec_list.append(rec)

        return rec_list

    def title_recommend(self, title, min_relevancy=0.3):
        """Recommends papers based on a title.

        The basic algorithm here is scikit-learn's tf-idf model.  The model is automatically fit to the set of calls
        when a CallInstance Object is initialized.  This is functionally equivalent to abstract_recommend ignoring
        some slight optimization in tokenization/lemmatization.

        ...

        Parameters
        ----------
        title : str
            a string representing a title

        min_relevancy : float
            the minimum relevancy required for a call to appear in the results

        Returns
        -------
        rec_list : list of CallRec
            a list of calls with relevancy greater than min_relevancy and containing additional information (see below)

        CallRec Specifications
        ----------------------
        criteria : str
            a string, equivalent to the abstract parameter

        rec_info : list of str
            brief summary of the abstract used to perform the search
        """

        # A list of CallRec Objects, to be returned by the function
        rec_list = []

        # Associates each call with a relevance score
        words = []
        tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
        for word in tokenizer.tokenize(title):
            if word not in words and word not in STOPWORDS and word.isalpha():
                words.append(word)

        index_zip = zip(self.calls, self.relevance(words))

        # Finds sufficiently relevant calls and places the resulting CallRec object in rec_list
        for call, rel in index_zip:
            if rel > min_relevancy:

                # Makes a brief summary
                ret_str = "Based on the abstract beginning title " + title + "."
                rec = CallRec(call, rel, "title", words, ret_str)
                rec_list.append(rec)

        return rec_list
