import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process


st.markdown("<h1 style='text-align:center'>Bank Category Prediction</h1>", unsafe_allow_html=True)


search_df = pd.read_csv("search_data.csv")

# Initialize the session as none
if 'test_df' not in st.session_state:
    st.session_state['test_df'] = None


# Upload file to be predicted
st.write("#### Upload dataset to be predicted")
test_file = st.file_uploader(label="", type=[".csv"], key=2)


if test_file is not None:
    
    # Prep the test file
    test_df = pd.read_csv(test_file)[['Transaction Details']].dropna()
    test_df['Category'] = ""
    test_df['Fuzz Score'] = ""
    test_df['Closest Match'] = ""

    
    if st.button("Make Prediction"):

        # Parse the search dataframe into a dictionary
        search_choices = dict(zip(search_df.index, search_df["Transaction Details"]))

 
        progress_bar = st.progress(0.0)
        progress_lenght = test_df.shape[0]


        for i, row in test_df.iterrows():

            fuzz_query = row['Transaction Details']
            
            # TODO Replace all '/' and numerical characters in descriptions with empty spaces and perform token_set_ratio
            # MATCHES: dict.values
            # RETURNS: dict_value, score, dict_key
            extracted = process.extractOne(query=fuzz_query, choices=search_choices, scorer=fuzz.partial_ratio)
            
            # Append results
            row['Category'] = search_df.at[int(extracted[2]), 'Category']
            row['Fuzz Score'] = extracted[1]
            row['Closest Match'] = extracted[0]

            
            # Update progress bar
            progress = round((i/progress_lenght), 1)
            progress_bar.progress(progress)
        
        progress_bar.progress(1.0)
        
        # Cache the data
        st.session_state['test_df'] = test_df

        st.success("Prediction Completed!")


if st.session_state['test_df'] is not None:
    
    fuzz_threshold = st.number_input(label="Fuzz Score Threshold", min_value=0, max_value=100, value=90)

    display_df = st.session_state['test_df']
    display_df = display_df[display_df['Fuzz Score'] >= fuzz_threshold]

    # Download file
    def get_download_data():
        download_df = display_df[['Transaction Details', 'Category']]
        return download_df.to_csv(index=False).encode('utf-8')

    st.download_button(label="Download Results", data=get_download_data(), file_name="Predicted Bank Data.csv")


    # Display dataframe
    st.dataframe(pd.DataFrame(display_df))




# If user wishes to add new data
with st.expander("Add to existing data"):
    new_files = st.file_uploader(label="", type=[".csv"], key=1, accept_multiple_files=True)
    
    if new_files:

        with st.spinner("Processing files..."):

            # Read new data and add it to existing data
            new_dfs = [pd.read_csv(file)[['Transaction Details', 'Category']] for file in new_files]
            new_dfs.append(search_df)
            search_df = pd.concat(new_dfs)

            # Clean data
            search_df.dropna(inplace=True)
            search_df.drop_duplicates(subset=['Transaction Details'], inplace=True)

        if st.button("Add"):
            search_df.to_csv("search_data.csv", index=False)
            st.success("New data saved successfully!")
