import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import sketch, rules, dm, art

st.title("Sketch & Spell")

if "hero" not in st.session_state:  # runs once, at the start
    st.session_state.hero = {"class": "Bard", "stats": {"STR": -1, "DEX": 2, "CHA": 3}, "health": 4}
    st.session_state.scene = "A chasm splits the cavern floor. What do you do?"

st.write(st.session_state.scene)
canvas = st_canvas(stroke_width=4, stroke_color="#000000", background_color="#ffffff",
                   height=320, width=480, key="draw", return_image_data=True)

if st.button("Cast your drawing") and canvas.image_data is not None:
    image = Image.fromarray(canvas.image_data.astype("uint8")).convert("RGB")
    with st.spinner("The page is turning..."):
        guesses = sketch.recognise(image)                      # I. read the drawing
        drawing = guesses[0][0]
        result = rules.resolve(st.session_state.hero, drawing)  # II-V. dice and outcome
        story = dm.narrate(st.session_state.hero, st.session_state.scene, drawing, result)  # VI. tell
        picture = art.paint(story + ", ink and watercolour fantasy book illustration, warm candlelight")  # VI. paint
    st.caption(f"The Oracle sees: {guesses}  ·  {result}")
    st.image(picture)
    st.write(story)
    st.session_state.scene = story