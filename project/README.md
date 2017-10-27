# Search by Inference

### Project

#### Matthew A Snell

##### (Astro7467@GitHub)

#### GA Data Science, Singapore, Cohort #5

## Project Object:

- Use Ngram based indexing & search phrase x-referencing 
- Leverage TensorFlow Word2Vec to parse documents
- Vectorize (new) words
- Build related (inc. synomons, typos, alternative spellings, or associated ) relationships based on document usage of words
  - Eliminates need for stemming (in theory and with a large/quality enough corpus)
- Compare Ngram only vs Ngram extension via Word2Vec generated similarities

## High-Level Flow:

### Indexing:
- Scan document corpus for new/unindexed documents
- For each new document;
  - Load/Extract Text
  - Remove STOP words
  - Vectorize
  - Add document reference (path/url) to Search Index (UID for each doc/source?)
    - Count occurances of each word/Ngram in document
    - Add counts to index (TF vector for word) against source UID
    
### Searching:
- Score each UID source/doc
  - Primary score - Total count/occurances of each search word/Ngram
  - Secondary Score - Total Count/occurnaces of each with 1st/Highest Similarity words from W2V added to search terms
- Present Top 10 Direct Primary Score vs Top 10 Secondary Scores

### Word2Vec Model:
- Built on the word2vec_basic.py code currently
- Run in batch mode
- Saves Top 8 most similiar words for use in Search
- Can update at will, model will just get better if there are more documents but newly indexed documents still benefit from past training due to ngram usage

See ./presentation/Presentation-SWI.{ipynb, slides.html} for some additional info

## Notes
- Project is a proof-of-concept for a personal project (see Picses/PerSea repository)
- Leveraged as a Python 2 then 3 learning experience
  - lots of PEP8 issues :/
- Currently only processes TEXT (.txt) and CSV (.csv) files
- Initial Ngram indexing currently is very slow
- Currently uses shelves - so datafiles can get large eg multi GB for a corpus <200Mb
- As shelves must have a string key at the root
  - vectors are subkeyed
  - is unusually large
  - is code ugly e.g. `dbstores['vectors']['vectors'][<vector>]`
- The raw output of Word2Vec is not stored - too large for shelves with <100k words/vectors
  - Instead similiarities are extracted and stored with a 'word' key
- Tensorflow CPU vs GPU seems to make little difference to run speed (approx 1.5hrs per 1000 steps)
- Yeah, I know, the code lacks 'class'
- Developed envronment;
  - anaconda3
    - Started in anaconda2
  - PyCharm (hence knowledge of PEP8 issues)
    - Originally started with spyder3, then atom
  - Ubuntu MATE 16.04 (Desktop) & 17.04 / 17.10 (Laptop)
  

## Running

*Note; by default the program is in trace mode and will dump a lot of data to the screen - see `_logSysLevel`*

- Directory Scructure;

  `./data`          - where the shelve files will be created
  
  `./data/srcdata`  - Currently only the normalised (stop words and non-alphanumerics removed) are stored
  
  `./corpus`        - files to be indexed should be placed here

- 1st run;
  
  > `./SearchWithInference.py --load-stopwords`
  
  Will load the `stopwords.txt` file into a shelve file (other empty shelves will also be created)
  
- Indexing files;
  
  > `./SearchWithInference.py --index`
  
  Will parse the files in `./corpus` - currently very slow
  
- Data storage stats;
  
  > `./SearchWithInference.py --stats` 

  Will dump a series of data points on (most) of the datastores / shelves
  

- Searching;
  
  > `./SearchWithInference.py --search "phrase/words I am searching for"`
  
  Will return best 10 matches (scored and with Ngram match counts)
  
  Will use similarity data if present for 2nd set of results

- Word2Vector Similiarty Data Building;
  
  > `./SearchWithInference.py --w2v`
  
  Will build a scratch shelve and start running Tensorflow Word2Vec
  
  Scratch shelve can get large, eg 2.5GB+ for <200MB corpus
  
  Will extract every known vector from the model after running
  
 

Example Results;

> `./SearchWithInference.py --search "data science python pdf"`


trimmed output inc. trace level output 

**1st pass - ngram only**

```
Search - Search Words;  Length 4, Type <class 'list'> -
         data, pdf, python, science
:
:
--------------------------------------------------------------------------------
topSrcMatches:
   1 : FILE TXT 217f07c7-0f4f-5960-9777-299eaaa403a6 1.0                      11 ./corpus/text/Library_List.txt
   2 : FILE TXT 54b33b50-f2a8-53a9-a5b0-8bdba164926f 0.14126269165580668       5 ./corpus/text/text8-lines.txt
   3 : FILE CSV 2fb4c755-533d-5f91-9738-a6f623205358 0.1328579928743543        5 ./corpus/csv/deep-nlp-Sheet_2.csv
   4 : FILE CSV 423406bc-7e7c-51a3-b8d3-442dced8fc2e 0.005597581480968173      1 ./corpus/csv/stanford-natural-language-inference-corpus_1.0_dev.csv
   5 : FILE CSV 5ab6f8d5-b65c-589e-84fb-c6fedc80d60c 0.004579839393519414      1 ./corpus/csv/stanford-natural-language-inference-corpus_1.0_test.csv
   6 : FILE TXT 3981177d-fe7a-5f64-9404-e76afb2fa451 0.002576469572288563      2 ./corpus/text/core-wordnet.txt
   7 : FILE TXT f6eaf116-73be-5f7a-aa27-3fb6b7198a8e 0.0005469954457246669     3 ./corpus/text/dictionary.txt
   8 : FILE TXT 95c45ba6-330b-5887-b03c-ec07f313fd15 0.0005134588085339031     2 ./corpus/text/t8.shakespeare.txt
   9 : FILE TXT 37c2d728-85ce-51d6-ae62-dcc51d96da19 0.0003438351272924433     2 ./corpus/text/i11.txt
  10 : FILE TXT 7fe89367-9b23-56a8-9b49-74bb1ab9f5dc 0.00022180653441359027    1 ./corpus/text/LICENSE-README.txt
--------------------------------------------------------------------------------
:
:
Search - ngrams used;  Length 12, Type <class 'list'> -
         data, data science, data science python, pdf, python, python data,
         python data science, python pdf, science, science data, science pdf, science python
         
Search - ngrams not used;  Length 28, Type <class 'list'> -
         data pdf, data pdf python, data pdf science, data python, data python pdf,
         data python science, data science pdf, pdf data, pdf data python, pdf data science,
         pdf python, pdf python data, pdf python science, pdf science, pdf science data,
         pdf science python, python data pdf, python pdf data, python pdf science,
         python science, python science data, python science pdf, science data pdf, science data python, 
         science pdf data, science pdf python, science python data, science python pdf
```

**2nd pass - extended ngram with Word2Vec similar words added**

```
Search; Initiated Similarity Extension Search
Search - Add Similiar Word; heapst
Search - Add Similiar Word; vetoed
Search - Add Similiar Word; ipynb
Search - Add Similiar Word; paralleld
Search - Search Words;  Length 8, Type <class 'list'> -
         data, heapst, ipynb, paralleld, pdf, python, science, vetoed
:
:
--------------------------------------------------------------------------------
topSrcMatches:
   1 : FILE TXT 217f07c7-0f4f-5960-9777-299eaaa403a6 1.0                      13 ./corpus/text/Library_List.txt
   2 : FILE TXT 54b33b50-f2a8-53a9-a5b0-8bdba164926f 0.14138024553093573       6 ./corpus/text/text8-lines.txt
   3 : FILE CSV 2fb4c755-533d-5f91-9738-a6f623205358 0.12918902995888712       5 ./corpus/csv/deep-nlp-Sheet_2.csv
   4 : FILE CSV 423406bc-7e7c-51a3-b8d3-442dced8fc2e 0.005443140901718332      1 ./corpus/csv/stanford-natural-language-inference-corpus_1.0_dev.csv
   5 : FILE CSV 5ab6f8d5-b65c-589e-84fb-c6fedc80d60c 0.004453478919587727      1 ./corpus/csv/stanford-natural-language-inference-corpus_1.0_test.csv
   6 : FILE TXT 3981177d-fe7a-5f64-9404-e76afb2fa451 0.00250535072253251       2 ./corpus/text/core-wordnet.txt
   7 : FILE TXT f6eaf116-73be-5f7a-aa27-3fb6b7198a8e 0.0012003796939820386     4 ./corpus/text/dictionary.txt
   8 : FILE TXT 95c45ba6-330b-5887-b03c-ec07f313fd15 0.0007703937903816965     4 ./corpus/text/t8.shakespeare.txt
   9 : FILE TXT 37c2d728-85ce-51d6-ae62-dcc51d96da19 0.00033434386554915364    2 ./corpus/text/i11.txt
  10 : FILE TXT 7fe89367-9b23-56a8-9b49-74bb1ab9f5dc 0.00021743454224397768    1 ./corpus/text/LICENSE-README.txt
--------------------------------------------------------------------------------
:
:
Search - ngrams used;  Length 17, Type <class 'list'> -
         data, data ipynb, data science, data science python, heapst, ipynb, paralleld, pdf, python, 
         python data, python data science, python pdf, science, science data, science pdf, science python, vetoed

Search - ngrams not used;  Length 383, Type <class 'list'> -
         data heapst, data heapst ipynb, data heapst paralleld, data heapst pdf, data heapst python, 
         data heapst science, data heapst vetoed, data ipynb heapst, data ipynb paralleld, data ipynb pdf,
         data ipynb python, data ipynb science, data ipynb vetoed, data paralleld, data paralleld heapst,
         data paralleld ipynb, data paralleld pdf, data paralleld python, data paralleld science, data paralleld vetoed, 
         data pdf, data pdf heapst, data pdf ipynb, data pdf paralleld, data pdf python, data pdf science, data pdf vetoed,
         data python, data python heapst,..... 
```

*`ngrams not used` refers to Ngrams not seen/indexed before*
