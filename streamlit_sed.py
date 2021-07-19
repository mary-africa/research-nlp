import streamlit as st

from experimental.sed.text.utils.pipeline import Inference

#inference tool
infer = Inference('language model')

# this is the main function in which we define our webpage 
def main():
      # giving the webpage a title
    st.title("Experimental MaryNLP Tools")
      
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed
    html_temp = """
    <div style ="padding:13px">
    <h1 style ="color:green;text-align:center;">Mary Text Tools</h1>
    </div>
    """
      
    # this line allows us to display the front end aspects we have 
    # defined in the above code
    st.markdown(html_temp, unsafe_allow_html = True)
      
    # the following lines create text boxes in which the user can enter 
    # the data required to make the prediction
    tokenization_text = st.text_input("Text to be tokenized", "maria anawezekana kuwa mtu mwenye Kiswahili kizuri kuliko wote")
    result =""

    if st.button("Tokenize"):
        result = infer.tokenize(tokenization_text)
    st.success(result)

    example_word = st.text_input("Example Word", "atacheza")
    example_analogy = st.text_input("Example Analogy", "alicheza")
    word = st.text_input("Test Word", "ataenda")
    result =""

    if st.button("Get Analogy"):
        result = infer.word_analogy(example_word, example_analogy, word)
    st.success(result)

    context = st.text_input("Context", "hapana siwezi kwenda nipe mda kabla ya")
    # result =""

    # if st.button("Predict Next Word"):
    next_word = infer.get_preds(context)
    st.success(next_word)
      
    # the below line ensures that when the button called 'Predict' is clicked, 
    # the prediction function defined above is called to make the prediction 
    # and store it in the variable result
     
if __name__=='__main__':
    main()