###################################################################################################################
#                                                   LOAD PACKAGES                                                      #
########################################################################################################################
# <editor-fold desc="Loading packages">
import re
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import string

from PIL import Image
from nltk import pos_tag, ne_chunk
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import SpaceTokenizer, TabTokenizer, LineTokenizer, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# </editor-fold>

########################################################################################################################
#                                                   FUNCTION AREA                                                      #
########################################################################################################################

# *************************************************
# *****      get_word_net_pos_ta              *****
# *************************************************
# <editor-fold desc=" get_word_net_pos_tag Map pos tag to the first character, a function that is used in lemmatize ">
def get_word_net_pos_tag(word):
    tag_in_word = nltk.pos_tag([word])[0][1][0].upper()
    tag_dictionary = {'J': wordnet.ADJ,
                      'N': wordnet.NOUN,
                      'V': wordnet.VERB,
                      'R': wordnet.ADV}
    return tag_dictionary.get(tag_in_word, wordnet.NOUN)


# </editor-fold>

# *************************************************
# *****      text_pre_processing              *****
# *************************************************
# <editor-fold desc="Data Prep-processing ">
def text_pre_processing(text, remove_number=True, stop_word=True, stop_word_language='english',
                        remove_punctuation=True):
    # ---------------------------------------------
    # Patterns
    results_chunk = ''
    results_named_entitiy = ''

    patterns1 = r'@[A-Za-z0-9_]+'
    pattterns2 = r'https?://[^ ]+'
    combined_patterns = r'|'.join((patterns1, pattterns2))
    www_patterns = r'www.[^ ]+'
    negations_dic = {"isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not",
                     "haven't": "have not", "hasn't": "has not", "hadn't": "had not", "won't": "will not",
                     "wouldn't": "would not", "don't": "do not", "doesn't": "does not", "didn't": "did not",
                     "can't": "can not", "couldn't": "could not", "shouldn't": "should not", "mightn't": "might not",
                     "mustn't": "must not"}
    negations_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

    # ---------------------------------------------
    # convert to lower case
    results = str(text)

    # ---------------------------------------------
    # Text Cleaning
    results = re.sub(combined_patterns, '', results)
    results = re.sub(www_patterns, '', results)
    results = results.lower()
    results = negations_pattern.sub(lambda x: negations_dic[x.group()], results)
    results = re.sub("[^a-zA-Z]", " ", results)

    results = results.replace("(<br/>)", "")
    results = results.replace('(<a).*(>).*(</a>)', '')
    results = results.replace('(&amp)', '')
    results = results.replace('(&gt)', '')
    results = results.replace('(&lt)', '')
    results = results.replace('(\xa0)', ' ')

    # ---------------------------------------------
    if (remove_number) & (results != ''):
        results = re.sub(r'\d+', '', results)

    # ---------------------------------------------
    if remove_punctuation & (results != ''):
        translator = str.maketrans('', '', string.punctuation)
        results = results.translate(translator)

    # ---------------------------------------------
    # Remove whitespaces
    results = results.strip()

    # ---------------------------------------------
    # Line Tokenize
    if results != '':
        line_tokenizer = LineTokenizer()
        results = line_tokenizer.tokenize(results)
        results = list(filter(None, results))
        results = results[0]

    # ---------------------------------------------
    # Tab Tokenize
    if results != '':
        tab_tokenizer = TabTokenizer()
        results = tab_tokenizer.tokenize(results)
        results = list(filter(None, results))
        results = results[0]

    # ---------------------------------------------
    # Space Tokenizer
    if results != '':
        space_toknizer = SpaceTokenizer()
        results = space_toknizer.tokenize(results)
        results = list(filter(None, results))
        results = ' '.join([w for w in results])

    # -----------------------------------------------
    # Lemmatization using NLTK
    if results != '':
        lemmatizer_of_text = WordNetLemmatizer()
        word_list = word_tokenize(results)
        results = ' '.join([lemmatizer_of_text.lemmatize(w, get_word_net_pos_tag(w)) for w in word_list])

    # ---------------------------------------------
    # Stemming using NLTK
    if results != '':
        stemmer = PorterStemmer()
        if type(results) == list:
            results = ' '.join(str(w) for w in results)
        results = word_tokenize(str(results))
        results = [stemmer.stem(word) for word in results]
        results = ' '.join(str(w) for w in results)

    # ---------------------------------------------
    # Remove Stop Words
    if stop_word & (results != ''):
        nltk.download('stopwords')
        stop_words = set(stopwords.words(stop_word_language))
        word_tokens = word_tokenize(results)
        results = ' '.join(str(w) for w in word_tokens if not w in stop_words)

    # ---------------------------------------------
    # Chunking of the input, will be used ofr coloring of the text
    if results != '':
        result_str = TextBlob(results)
        reg_exp = 'NP: { < DT >? < JJ > * < NN >}'
        rp = nltk.RegexpParser(reg_exp)
        results_chunk = rp.parse(result_str.tags)
    # results_chunk.draw()

    # ---------------------------------------------
    # Named Entity Recognition
    if results != '':
        results_named_entitiy = ne_chunk(pos_tag(word_tokenize(results)))

    return results, results_chunk, results_named_entitiy

# </editor-fold>


# *************************************************
# *****      get_ngram_word_freq              *****
# *************************************************
# <editor-fold desc=" get_ngram ">
def get_ngram_word_freq(text, ngram=None, apply_stop_words=True):
    if (ngram == None) & (apply_stop_words == False):  # Return word frequency
        vector = CountVectorizer().fit(text)
    elif (ngram == None) & (apply_stop_words == True):  # Return word frequency with applying stop words
        vector = CountVectorizer(stop_words='english').fit(text)
    elif (ngram == (2, 2)) & (apply_stop_words == False):  # Return bigram word frequency without applying stop words
        vector = CountVectorizer(ngram_range=(2, 2)).fit(text)
    elif (ngram == (3, 3)) & (apply_stop_words == False):  # Return trigram word frequency without applying stop words
        vector = CountVectorizer(ngram_range=(3, 3)).fit(text)
    elif (ngram == (2, 2)) & (apply_stop_words == True):  # Return bigram word frequency with applying stop words
        vector = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(text)
    elif (ngram == (3, 3)) & (apply_stop_words == True):  # Return trigram word frequency with applying stop words
        vector = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(text)

    bag_of_words = vector.transform(text)

    sum_words_in_text = bag_of_words.sum(axis=0)
    ngram_words_freq_in_text = [(word, sum_words_in_text[0, index]) for word, index in vector.vocabulary_.items()]
    ngram_words_freq_in_text = sorted(ngram_words_freq_in_text, key=lambda x: x[1], reverse=True)
    ngram_words_freq_in_text = pd.DataFrame(ngram_words_freq_in_text, columns=['sentence', 'count_sentence'])
    ngram_words_freq_in_text['count_sentence'] = min_max_zero_one_norm(ngram_words_freq_in_text[['count_sentence']])
    ngram_words_freq_in_text = ngram_words_freq_in_text.set_index('sentence')

    return ngram_words_freq_in_text


# </editor-fold>


# *************************************************
# *****      get_ngram_word_tfidf             *****
# *************************************************
# <editor-fold desc=" get_ngram_word_tfidf ">
def get_ngram_word_tfidf(text, ngram=None, apply_stop_words=True):
    # instantiate CountVectorizer()
    cv = CountVectorizer()
    if (ngram == None) & (apply_stop_words == False):  # Return word frequency
        vector = cv.fit(text)
    elif (ngram == None) & (apply_stop_words == True):  # Return word frequency with applying stop words
        vector = CountVectorizer(stop_words='english').fit(text)
    elif (ngram == (2, 2)) & (apply_stop_words == False):  # Return bigram word frequency without applying stop words
        vector = CountVectorizer(ngram_range=(2, 2)).fit(text)
    elif (ngram == (3, 3)) & (apply_stop_words == False):  # Return trigram word frequency without applying stop words
        vector = CountVectorizer(ngram_range=(3, 3)).fit(text)
    elif (ngram == (2, 2)) & (apply_stop_words == True):  # Return bigram word frequency with applying stop words
        vector = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(text)
    elif (ngram == (3, 3)) & (apply_stop_words == True):  # Return trigram word frequency with applying stop words
        vector = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(text)

    bag_of_words = vector.transform(text)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    # tf-idf scores
    tf_idf_vector = tfidf_transformer.fit_transform(bag_of_words)
    feature_names = vector.get_feature_names()
    # get tfidf vector for first document
    document_vector = pd.DataFrame(tf_idf_vector.T.todense(), index=feature_names)
    document_vector_tfidf = pd.DataFrame(document_vector.mean(axis=1), columns=["count_sentence"])
    document_vector_tfidf = document_vector_tfidf.sort_values(by=["count_sentence"], ascending=False)
    document_vector_tfidf['count_sentence'] = min_max_zero_one_norm(document_vector_tfidf[['count_sentence']])

    return document_vector_tfidf


# </editor-fold>


# *************************************************
# *****      word_cloud_plot_save             *****
# *************************************************
# <editor-fold desc=" word_cloud_plot_save ">
def word_cloud_plot_save(reviews_text, id=None, plot_calculation_type='freq', bigram_include=True, trigram_include=True,
                         font_path='C:/Windows/Fonts/timesbd.ttf', path_to_save='', save_file_name='_WordCloud',
                         background_color='white'
                         , height=800, width=1200, colormap='tab10', collocations=False, img_show=True,
                         mask_image_name='', contour_width=2, contour_color='black'):
    if plot_calculation_type == 'freq':
        onegram_reviews = get_ngram_word_freq(text=reviews_text, ngram=None, apply_stop_words=True)
        ngram_reviews = onegram_reviews
        if bigram_include:
            bigram_reviews = get_ngram_word_freq(text=reviews_text, ngram=(2, 2), apply_stop_words=True)
            ngram_reviews = pd.concat([ngram_reviews, bigram_reviews])
        if trigram_include:
            trigram_reviews = get_ngram_word_freq(text=reviews_text, ngram=(3, 3), apply_stop_words=True)
            ngram_reviews = pd.concat([ngram_reviews, trigram_reviews])
    elif plot_calculation_type == 'tfidf':
        onegram_reviews = get_ngram_word_tfidf(text=reviews_text, ngram=None, apply_stop_words=True)
        ngram_reviews = onegram_reviews
        if bigram_include:
            bigram_reviews = get_ngram_word_tfidf(text=reviews_text, ngram=(2, 2), apply_stop_words=True)
            ngram_reviews = pd.concat([ngram_reviews, bigram_reviews])
        if trigram_include:
            trigram_reviews = get_ngram_word_tfidf(text=reviews_text, ngram=(3, 3), apply_stop_words=True)
            ngram_reviews = pd.concat([ngram_reviews, trigram_reviews])

    ngram_reviews['count_sentence'] = ngram_reviews['count_sentence'].astype(float)

    if mask_image_name == '':
        word_cloud = WordCloud(background_color=background_color, height=height, width=width, colormap=colormap,
                               collocations=collocations,
                               font_path=font_path).generate_from_frequencies(ngram_reviews.count_sentence)
    else:
        mask_png = np.array(Image.open(mask_image_name))

        # Create a word cloud image
        word_cloud = WordCloud(background_color='white', height=mask_png.shape[0], width=mask_png.shape[1],
                               contour_width=contour_width, contour_color=contour_color, mask=mask_png,
                               collocations=collocations,
                               font_path=font_path).generate_from_frequencies(
            ngram_reviews.count_sentence)
        image_colors = ImageColorGenerator(mask_png)
        word_cloud.recolor(color_func=image_colors)
    # store to file
    if id == None:
        word_cloud.to_file(path_to_save + plot_calculation_type + save_file_name + '.png')
    else:
        word_cloud.to_file(path_to_save + str(id) + plot_calculation_type + save_file_name + '.png')
    # Display the generated image:
    if img_show:
        plt.figure(figsize=[20, 10])
        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    return

# </editor-fold>

# *************************************************
# *****      min_max_zero_one_norm            *****
# *************************************************
# <editor-fold desc=" MinMAxZeroOneNorm Normalization ">
def min_max_zero_one_norm(data):
    scaler = MinMaxScaler()
    data_min_max_zero_one = scaler.fit_transform(data)
    return data_min_max_zero_one


# </editor-fold>

# *************************************************
# *****      sentiment_analysis               *****
# *************************************************
# <editor-fold desc=" Sentiment Analysis ">
def sentiment_analysis(sentence):
    nltk_sentiment = SentimentIntensityAnalyzer()
    score = nltk_sentiment.polarity_scores(sentence)
    return score


# </editor-fold>

# *************************************************
# *****      apply_sentiment_analysis         *****
# *************************************************
# <editor-fold desc=" Apply Sentiment Analysis ">
def apply_sentiment_analysis(df, comment_col_name='preprocessed_comment_text'):
    sentiment_results = pd.DataFrame([sentiment_analysis(row) for row in df[comment_col_name]])
    df['sentiment'] = sentiment_results['compound'].apply(lambda x: 'Positive' if x >= 0 else 'Negative')
    return df


# </editor-fold>

########################################################################################################################
#                                                    MAIN CODE AREA                                                    #
########################################################################################################################

# *************************************************
# *****      Load data                        *****
# *************************************************
# <editor-fold desc=" Load data">
# Please download data from : https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews
path_to_save = 'D:/01_ABASADI/04_WordClod/04_Source_Code/03_Exemplary_Output/'
data_set = pd.read_csv(
    "D:/01_ABASADI/04_WordClod/01_DataSets/womens_ecommerce_clothing_reviews/Womens Clothing E-Commerce Reviews.csv",
    index_col=0)

print("There are {} observations and {} features in this dataset. \n".format(data_set.shape[0], data_set.shape[1]))
print("There are {} clothes".format(len(data_set['Clothing ID'].unique())))
# </editor-fold>
# *************************************************
# *****            Data Preprocessing         *****
# *************************************************
# <editor-fold desc="Data Preprocessing">
# -------------------------------------------------
# 1. delete null reviews
data_set.drop('Title', axis=1, inplace=True)
data_set = data_set[~data_set['Review Text'].isnull()]
print("There are {} clothes after removing nulls ".format(len(data_set['Clothing ID'].unique())))
data_set.columns = ['Clothing_ID', 'Age', 'Review_Text', 'Rating', 'Recommended_IND', 'Positive_Feedback_Count',
                    'Division_Name', 'Department_Name', 'Class_Name']

# -------------------------------------------------
# 2. Preprocessing
data_set['preprocessed_text_review'] = ''
for idx in data_set.index:
    print(idx)
    review_preprocessed, results_chunk, results_named_entitiy = text_pre_processing(text=data_set.Review_Text[idx],
                                                                                    remove_number=True, stop_word=True,
                                                                                    stop_word_language='english',
                                                                                    remove_punctuation=True)
    data_set.loc[idx, 'preprocessed_text_review'] = review_preprocessed
# </editor-fold>

# *************************************************
# *****            Visualization              *****
# *************************************************
# <editor-fold desc="Plotting">
for id in data_set.Clothing_ID.unique():
    id = 1078
    reviews_text = data_set.loc[data_set.Clothing_ID == id, 'preprocessed_text_review']
    word_cloud_plot_save(reviews_text)
    word_cloud_plot_save(reviews_text, plot_calculation_type='tfidf')
    word_cloud_plot_save(reviews_text, plot_calculation_type='tfidf', mask_image_name='')
    word_cloud_plot_save(reviews_text, plot_calculation_type='freq',
                     mask_image_name='D:/01_ABASADI/04_WordClod/04_Source_Code/02_Images/heart.png')
    word_cloud_plot_save(reviews_text, plot_calculation_type='tfidf',
                     mask_image_name='D:/01_ABASADI/04_WordClod/04_Source_Code/02_Images/heart.png')
# </editor-fold>

# *************************************************
# *****            Sentiment Data Analysis    *****
# *************************************************
# -------------------------------------------------
# Apply Sentiment Analysis
data_set = apply_sentiment_analysis(data_set, comment_col_name='preprocessed_text_review')

# *************************************************
# *****            Visualization              *****
# *************************************************
# plot Positive comments for clothing id=1078
id=1078
reviews_text = data_set.loc[
    (data_set.data_set.Clothing_ID == id) & (data_set['sentiment'] == 'Positive'), 'preprocessed_comment_text']
word_cloud_plot_save(reviews_text, path_to_save='',
                     save_file_name='PositiveComments')