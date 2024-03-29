import re
import pandas as pd
import streamlit as st
import pickle
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
import nmslib


def ngrams(string, n=3):
    """Takes an input string, cleans it and converts to ngrams"""

    string = str(string)
    string = fix_text(string)
    string = string.encode("ascii", errors="ignore").decode()

    chars_to_remove = ['(', ')', '[', ']', '{', '}', '|', '.', "'", '-', ':']
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('/', ' ')

    # get rid of multiple spaces and replace with a single
    string = re.sub(' +', ' ', string).strip()

    string = ' '+ string +' ' # pad names for ngrams...
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align:center'>Bank Category Prediction</h1>", unsafe_allow_html=True)

search_df = pd.read_csv("search_data.csv")
transac_detials = list(search_df['Transaction Details'].unique())


# Initialize the session as none
if 'test_df' not in st.session_state:
    st.session_state['test_df'] = None


# Upload file to be predicted
st.write("#### Upload dataset to be predicted")
st.info("Required columns are: Transaction Date, Amount, and Transaction Details")
test_file = st.file_uploader(label="", type=[".csv"], key=2)


if test_file is not None:
    
    # Prep the test file
    test_df = pd.read_csv(test_file)
    test_transac_detials = list(test_df['Transaction Details'].unique())

    
    if st.button('Make Prediction'):

        with st.spinner('Predicting..'):

            # Build the TFIDF off the clean dataset
            vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
            tf_idf_matrix = vectorizer.fit_transform(transac_detials)
            messy_tf_idf_matrix = vectorizer.transform(test_transac_detials)

            # Create a random matrix to index
            data_matrix = tf_idf_matrix

            # Intitialize the library, specify the space, the type of the vector and add data points 
            index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR) 

            index.addDataPointBatch(data_matrix)
            index.createIndex()

            # Use KNN to get similarity
            query_matrix = messy_tf_idf_matrix
            query_qty = query_matrix.shape[0]
            nbrs = index.knnQueryBatch(query_matrix, k=1, num_threads=4)


            # Add matched items
            matches = []

            for i in range(len(nbrs)):
                origional_name = test_transac_detials[i]

                try:
                    matched_name = transac_detials[nbrs[i][0][0]]
                    matched_category = search_df[search_df['Transaction Details']==matched_name].iat[0,3]
                    conf = nbrs[i][1][0]
                
                except:
                    matched_name = 'no match found'
                    matched_category = 'no category'
                    conf = None

                matches.append([origional_name, matched_name, matched_category, conf])

            matches = pd.DataFrame(matches, columns=['Transaction Details', 'Closest Match', 'Category', 'Match Confidence'])
            # Normalize and invert confidence column
            matches['Match Confidence'] = (matches['Match Confidence'].max() - matches['Match Confidence']) / (matches['Match Confidence'].max() - matches['Match Confidence'].min())
            results = test_df.merge(matches)

        
        # Cache the data
        st.session_state['test_df'] = results
        st.success("Prediction Completed!")


if st.session_state['test_df'] is not None:

    display_df = st.session_state['test_df']
    
    conf_threshold = st.number_input(
        label='Confidence Threshold',
        min_value = 0.00,
        max_value = 1.00,
        value = 0.80,
        step = 0.01
    )

    display_df = display_df[display_df['Match Confidence'] >= conf_threshold]
    display_df.reset_index(drop=True, inplace=True)

    # Download file
    def get_download_data():
        return display_df.drop(columns=['Closest Match']).to_csv(index=False).encode('utf-8')

    st.download_button(label="Download Results", data=get_download_data(), file_name="Predicted Bank Data.csv")

    # Display dataframe
    st.dataframe(pd.DataFrame(display_df))
    st.write(f"Number of rows: {display_df.shape[0]}")




# To add to existing search data
with st.expander("Add to Search Data"):
    st.info("Required columns are: Transaction Date, Transaction Details, CRDR, and Category")
    new_files = st.file_uploader(label="", type=[".csv"], key=1, accept_multiple_files=True)
    
    if new_files:

        with st.spinner("Processing files..."):

            # Read new data and add it to existing data
            new_dfs = [pd.read_csv(file)[['Transaction Date', 'Transaction Details', 'CRDR', 'Category']] for file in new_files]
            new_dfs.append(search_df)
            search_df = pd.concat(new_dfs)

            # Clean data
            search_df.dropna(inplace=True)
            #search_df.drop_duplicates(subset=['Transaction Details', 'Category'], inplace=True)
            search_df.drop_duplicates(inplace=True)

        if st.button("Add"):
            search_df.to_csv('search_data.csv', index=False)
            st.success('New data saved successfully!')
