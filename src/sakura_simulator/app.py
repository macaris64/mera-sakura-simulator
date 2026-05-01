"""Streamlit UI page for the SAKURA-II simulator."""

import datetime

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
            col1, col2, col3 = st.sidebar.columns(3)
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
            if col3.button(f"→ {entry.name}"):
                st.session_state["chat_model"] = entry.name
            col4, col5 = st.sidebar.columns(2)
            if col4.button(f"Download {entry.name}"):
                try:
                    path = registry.download(entry.name)
                    st.sidebar.success(f"Downloaded: {path.name}")
                except (ValueError, FileNotFoundError) as exc:
                    st.sidebar.error(f"Download failed: {exc}")
            if col5.button(f"Remove {entry.name}"):
                try:
                    registry.remove(entry.name)
                    st.sidebar.success(f"Removed: {entry.name}")
                except (ValueError, FileNotFoundError) as exc:
                    st.sidebar.error(f"Remove failed: {exc}")
    except FileNotFoundError:
        st.sidebar.warning("No model manifest found.")


def _render_chat_panel(model_name: str) -> None:
    """Render a WhatsApp-style chat interface for the given model."""
    from sakura_simulator.registry import ModelRegistry

    st.subheader(f"Chat — {model_name}")
    if st.button("× Close"):
        st.session_state["chat_model"] = None
        return

    history_key = f"chat_history_{model_name}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for msg in st.session_state[history_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            st.caption(msg["time"])

    user_input = st.chat_input("Type a message...")
    if user_input is not None:
        now = datetime.datetime.now().strftime("%H:%M")
        st.session_state[history_key].append(
            {"role": "user", "content": user_input, "time": now}
        )
        try:
            registry = ModelRegistry()
            entry = registry.get_model(model_name)
            if entry is None:
                raise ValueError(f"Model '{model_name}' not found in registry")
            from sakura_simulator.runtime import MeraRuntime

            result = MeraRuntime().infer(entry, entry.artifact_dir, user_input)
            now = datetime.datetime.now().strftime("%H:%M")
            st.session_state[history_key].append(
                {"role": "assistant", "content": result.text, "time": now}
            )
        except (ValueError, FileNotFoundError) as exc:
            now = datetime.datetime.now().strftime("%H:%M")
            st.session_state[history_key].append(
                {"role": "assistant", "content": f"Error: {exc}", "time": now}
            )
        st.rerun()


def main():
    """Render the Streamlit page."""
    st.set_page_config(page_title="SAKURA-II Simulator", page_icon="🌸")
    st.title("SAKURA-II NPU Simulator")
    st.markdown("EdgeCortix MERA Framework — Hardware-Agnostic Hello World")
    _render_model_control_center()

    chat_model = st.session_state.get("chat_model")
    if chat_model:
        _render_chat_panel(chat_model)

    if st.button("Activate Engine"):
        engine = _get_engine()
        st.success(engine.greeting())


if __name__ == "__main__":  # pragma: no cover
    main()
