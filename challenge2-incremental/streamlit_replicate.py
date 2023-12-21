import streamlit as st
import json
import replicate

# Function to read JSON and extract text
def get_caption_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data.get('captions', '')

# Streamlit app
def main():
    # Custom HTML for the title with "Web3" in red and red shadow
    title_html = """
    <style>
        .title {
            font-size: 40px; 
            font-weight: bold; 
            font-family: 'Arial', sans-serif; 
        }
        .web3 {
            color: #ff0000; /* Red color */
            text-shadow: 2px 2px #ff0000; /* Red shadow */
        }
        .recommenz {
            color: #0ff; /* Original color for the rest of the title */
            text-shadow: 2px 2px #0088ff; /* Blue shadow for 'Recommenz' */
        }
    </style>
    <h1 class="title"><span class="web3">Web3</span><span class="recommenz">Recommenz</span></h1>
    """
    st.markdown(title_html, unsafe_allow_html=True)

    # Load the caption from the JSON file
    caption = get_caption_from_json('recommend.json')

    # Button to display the caption and generate the video
    if st.button('Display Caption and Generate Video'):
        st.subheader("Caption:")
        st.write(caption)

        # Replicate's text2video API integration
        output = replicate.run(
            "pschaldenbrand/text2video:ed5518ac356d730caf25255a80e87c16823cf379398afbe26a7dc2e97cc58fee",
            input={"prompts": caption}
        )

        last_url = None
        for item in output:
            last_url = item

        if last_url:
            st.video(last_url)

if __name__ == "__main__":
    main()
