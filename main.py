from img_to_text import *
from story_generator import *
from text_to_speech import *

IMAGE_INPUT_PATH = "./data/images/"
AUDIO_OUTPUT_PATH = "./output/audio.wav"

load_dotenv()

def generate_ui():

    st.set_page_config(page_title = "Image to Story Generator")
    st.header("Convert image into story")
    uploaded_file = st.file_uploader("Choose an Image...", type = "jpg")
    
    if uploaded_file != None:
        bytes_data = uploaded_file.getvalue()
        with open(f"{IMAGE_INPUT_PATH}{uploaded_file.name}", "wb") as f:
            f.write(bytes_data)
        st.image(uploaded_file, caption = "Uploaded Image", use_column_width = True)

        #calling functions to generate story
        text = img_to_text(f"{IMAGE_INPUT_PATH}{uploaded_file.name}")

        with st.expander("Scenario"):
            st.write(text)

        story = create_story_using_gpt4_all_local(text)

        with st.expander("Story"):
            st.write(story)

        text_to_speech(story, AUDIO_OUTPUT_PATH)
        st.audio(AUDIO_OUTPUT_PATH)
        
if __name__ == "__main__":
    #text = img_to_text(IMAGE_INPUT_PATH)
    #story = create_story_using_gpt4_all_local(text)
    #text_to_speech(story, AUDIO_OUTPUT_PATH)
    generate_ui()
