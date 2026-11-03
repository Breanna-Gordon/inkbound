import io
import streamlit as st
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import sketch, rules, dm, art

CONFIDENT = 50   # below this %, ask the player what they drew
W, H = 720, 480  # same shape as the pictures (768 x 512)

st.title("Inkbound")

if "hero" not in st.session_state:  # runs once, at the start
    st.session_state.hero = {"class": "Bard", "stats": {"STR": -1, "DEX": 2, "CHA": 3}, "health": 4}
    st.session_state.scene = "A chasm splits the cavern floor. What do you do?"
    st.session_state.guesses = None
    st.session_state.last_turn = None
    st.session_state.turn_no = 0
    with st.spinner("Painting the first page..."):
        st.session_state.picture = art.paint("a deep rocky chasm in a cavern, a lone bard at the edge holding a lantern")

def strokes_only(image_data):
    """Keep just the player's lines, as black on white, for the Oracle."""
    alpha = Image.fromarray(image_data.astype("uint8"), "RGBA").getchannel("A")
    return ImageOps.invert(alpha).convert("RGB")

def play_turn(drawing):
    hero = st.session_state.hero
    with st.spinner("The page is turning..."):
        result = rules.resolve(hero, drawing)
        story, picture_text = dm.narrate(hero, st.session_state.scene, drawing, result)
        st.session_state.picture = art.paint(picture_text)
    st.session_state.last_turn = {"drawing": drawing, "result": result}
    st.session_state.scene = story
    st.session_state.guesses = None
    st.session_state.turn_no += 1  # gives the canvas a new key, so it clears

# The scene, with the player drawing on top of it
background = Image.open(io.BytesIO(st.session_state.picture)).convert("RGB").resize((W, H))
canvas = st_canvas(stroke_width=5, stroke_color="#ffe9a8", background_color="rgba(0,0,0,0)",
                   background_image=background, height=H, width=W,
                   key=f"draw-{st.session_state.turn_no}", return_image_data=True)

turn = st.session_state.last_turn
if turn:
    st.caption(f"You drew: {turn['drawing']}  ·  {turn['result']}")
st.write(st.session_state.scene)

if st.button("Cast your drawing") and canvas.image_data is not None:
    if canvas.image_data[:, :, 3].max() == 0:
        st.warning("Draw something on the scene first!")
    else:
        sketch_img = strokes_only(canvas.image_data)
        st.session_state.last_sketch = sketch_img
        guesses = sketch.recognise(sketch_img)
        if guesses[0][1] >= CONFIDENT:
            play_turn(guesses[0][0])
            st.rerun()
        else:
            st.session_state.guesses = guesses

if st.session_state.guesses:
    options = [f"{label} ({pct}%)" for label, pct in st.session_state.guesses]
    st.write("The Oracle isn't sure. What did you draw?")
    choice = st.radio("Pick one", options, label_visibility="collapsed")
    if st.button("That's it"):
        play_turn(st.session_state.guesses[options.index(choice)][0])
        st.rerun()

if "last_sketch" in st.session_state:
    with st.expander("What the Oracle saw"):
        st.image(st.session_state.last_sketch, width=240)