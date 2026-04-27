"""Streamlit UI page for the SAKURA-II simulator."""

import streamlit as st

from sakura_simulator.engine import SakuraEngine


@st.cache_resource
def _get_engine() -> SakuraEngine:
    return SakuraEngine()


def main():
    """Render the Streamlit page."""
    st.set_page_config(page_title="SAKURA-II Simulator", page_icon="🌸")
    st.title("SAKURA-II NPU Simulator")
    st.markdown("EdgeCortix MERA Framework — Hardware-Agnostic Hello World")

    if st.button("Activate Engine"):
        engine = _get_engine()
        st.success(engine.greeting())


if __name__ == "__main__":  # pragma: no cover
    main()
