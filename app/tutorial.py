"""Tutorial dialog for first-time users."""

import streamlit as st
from config import config


# Define tutorial steps
TUTORIAL_STEPS = [
    {
        "title": "Welcome!",
        "content": """

This quick tutorial will guide you through the main features of the app.

"""
    },
    {
        "content": """
The **left sidebar** is your control center for all waveform operations:

- Waveform type selection dropdown
- Waveform-specific parameters
- Precorrection toggle
- Ouput waveform
- Capture ADC samples

"""
    },
    {
        "content": """
Select your waveform type from the dropdown:

| Type | Description |
|------|-------------|
| **Serrodyne** | Multi-segment waveform with customizable frequencies |
| **Sine** | Simple sinusoidal waveform |
| **Sawtooth** | Linear ramp waveform |
| **Square** | Square wave with adjustable duty cycle |
| **Custom** | Upload your own waveform data |
"""
    },
    {
        "content": """
Each waveform type has different parameters:

**Serrodyne:**
- **Ratios** - Segment ratios (e.g., `1:5:3`)
- **Frequencies (MHz)** - One per segment; `0` for flat
- **Total time (µs)** - Duration of one period
- **Amplitude (LSB)** - Output amplitude

**Sine / Sawtooth / Square:**
- **Frequency (MHz)** - Signal frequency
- **Amplitude (LSB)** - Output amplitude
- **Duty Cycle** (square only) - High/low ratio
"""
    },
    {
        "content": """
When **precorrection** is enabled:

1. Outputs the original waveform
2. Captures the ADC response
3. Calculates a transfer function
4. Applies inverse correction
5. Outputs the corrected waveform

"""
    }
]


def show_tutorial():
    """Display the tutorial dialog for first-time users."""
    
    # Initialize step in session state
    if "tutorial_step" not in st.session_state:
        st.session_state.tutorial_step = 0
    
    @st.dialog("Welcome to Molecular Ions RFSoC Serrodyne", width="large")
    def tutorial_dialog():
        step = st.session_state.tutorial_step
        total_steps = len(TUTORIAL_STEPS)
        current = TUTORIAL_STEPS[step]
        
        # Step indicator
        st.progress((step + 1) / total_steps)
        st.caption(f"Step {step + 1} of {total_steps}")
        
        # Step title and content
        st.markdown(current["content"])
        
        st.divider()
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if step > 0:
                if st.button("← Previous", use_container_width=True):
                    st.session_state.tutorial_step -= 1
                    st.rerun()
        
        with col3:
            if step < total_steps - 1:
                if st.button("Next →", type="primary", use_container_width=True):
                    st.session_state.tutorial_step += 1
                    st.rerun()
            else:
                if st.button("✓ Finish", type="primary", use_container_width=True):
                    config.set("tutorial_done", True)
                    st.session_state.tutorial_step = 0
                    st.rerun()
    
    tutorial_dialog()
