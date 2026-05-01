"""Streamlit UI page for the SAKURA-II simulator."""

import streamlit as st

from sakura_simulator.engine import SakuraEngine


@st.cache_resource
def _get_engine() -> SakuraEngine:
    return SakuraEngine()


def _render_model_control_center() -> None:
    """Render the Model Control Center section in the sidebar."""
    from sakura_simulator.registry import ModelRegistry

    st.sidebar.header("Model Control Center")
    try:
        registry = ModelRegistry()
        models = registry.list_models()
        names = [m.name for m in models]
        selected = st.sidebar.selectbox("Active Model", names)
        st.session_state["active_model"] = selected
        for entry in models:
            ready = registry.is_space_ready(entry)
            indicator = ":green_circle:" if ready else ":red_circle:"
            st.sidebar.markdown(f"{indicator} {entry.name} v{entry.version}")
            compiled = registry.is_compiled(entry)
            compiled_indicator = ":green_circle:" if compiled else ":red_circle:"
            st.sidebar.markdown(f"{compiled_indicator} {entry.name} compiled")
            col1, col2 = st.sidebar.columns(2)
            if col1.button(f"Compile {entry.name}"):
                from sakura_simulator.compiler import MeraCompiler

                try:
                    path = MeraCompiler().compile(entry)
                    st.sidebar.success(f"Compiled: {path}")
                except (ValueError, FileNotFoundError) as exc:
                    st.sidebar.error(f"Compile failed: {exc}")
            if col2.button(f"Run {entry.name}"):
                from sakura_simulator.runtime import MeraRuntime

                try:
                    result = MeraRuntime().run(entry, entry.artifact_dir)
                    st.sidebar.info(f"Avg: {result.avg_latency_ms:.2f} ms")
                except (ValueError, FileNotFoundError) as exc:
                    st.sidebar.error(f"Run failed: {exc}")
    except FileNotFoundError:
        st.sidebar.warning("No model manifest found.")


def main():
    """Render the Streamlit page."""
    st.set_page_config(page_title="SAKURA-II Simulator", page_icon="🌸")
    st.title("SAKURA-II NPU Simulator")
    st.markdown("EdgeCortix MERA Framework — Hardware-Agnostic Hello World")
    _render_model_control_center()

    if st.button("Activate Engine"):
        engine = _get_engine()
        st.success(engine.greeting())


if __name__ == "__main__":  # pragma: no cover
    main()
