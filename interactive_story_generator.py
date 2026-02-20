# Interactive Story Generator - Single File
# Author: Abi (Abinaya Krishnamoorthy)
# Internship Project

# Step 1: Install Required Libraries
# Uncomment these lines if running in Google Colab
# !pip install transformers gradio

# Step 2: Load a Pre-trained Language Model
from transformers import pipeline, set_seed

# Load text generation pipeline
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

# Step 3: Define the Prompt-Based Story Generator Function
def generate_story_model(character_name, character_personality, story_theme):
    """
    Generate a story based on character name, personality, and story theme.
    """
    prompt = (
        f"Write a captivating {story_theme.lower()} story. "
        f"The story follows {character_name}, who is {character_personality}. "
        f"The adventure begins like this:\n"
    )
    
    story = generator(
        prompt,
        max_length=500,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.9
    )[0]['generated_text']
    
    return story

# Step 4: Build a Gradio Web Interface
import gradio as gr

def launch_gradio():
    interface = gr.Interface(
        fn=generate_story_model,
        inputs=[
            gr.Textbox(label="Character Name"),
            gr.Textbox(label="Personality Traits"),
            gr.Textbox(label="Story Theme (e.g., Fantasy, Romance, Mystery)")
        ],
        outputs="text",
        title="AI Story Generator",
        description="Generate stories based on a character name, personality, and theme using GPT-2."
    )
    return interface

# Launch the interface
interface = launch_gradio()
interface.launch(share=True)