import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the pre-trained GPT-2 model and tokenizer from Hugging Face
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Function to generate story based on prompt
def generate_story(prompt, max_length=200):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate story with the model
    output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    
    # Decode and return the generated text
    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

# Streamlit app UI
st.title("AI-Powered Story Generator")
st.subheader("Enter a prompt to generate a story!")

# Get user input for the prompt
user_prompt = st.text_input("Enter a theme or prompt:")

# Button to trigger story generation
if st.button("Generate Story"):
    if user_prompt:
        with st.spinner('Generating story...'):
            generated_story = generate_story(user_prompt)
        st.success("Story Generated!")
        st.write("### Generated Story:")
        st.write(generated_story)
    else:
        st.error("Please enter a prompt to generate the story.")
