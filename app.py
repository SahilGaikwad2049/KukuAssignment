"""PromoScout's responsive, local-only Streamlit workspace."""

import streamlit as st

from promoscout.dashboard.components import render_brand
from promoscout.dashboard.pages import (
    candidate_review_page,
    episode_explorer_page,
    exports_page,
    overview_page,
    rankings_page,
)
from promoscout.dashboard.composite_page import composite_promos_page
from promoscout.dashboard.styles import inject_styles
from promoscout.dashboard.series_workspace import series_workspace_page


def main() -> None:
    st.set_page_config(
        page_title="PromoScout · Kuku",
        page_icon="🎬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles(st)
    with st.sidebar:
        render_brand(st, sidebar=True)
        st.caption("A local workspace for screenplay intelligence and promo review.")
        st.markdown("---")
        st.caption("All screenplay data stays in this project workspace.")

    pages = {
        "Workspace": [
            st.Page(overview_page, title="Overview", icon=":material/dashboard:", default=True),
            st.Page(episode_explorer_page, title="Episode Explorer", icon=":material/movie:"),
            st.Page(candidate_review_page, title="Candidate Review", icon=":material/rate_review:"),
            st.Page(rankings_page, title="Rankings", icon=":material/leaderboard:"),
            st.Page(exports_page, title="Exports", icon=":material/download:"),
        ],
        "More": [
            st.Page(series_workspace_page, title="Series Workspace", icon=":material/upload_file:"),
            st.Page(composite_promos_page, title="Composite Promos", icon=":material/movie_filter:"),
        ],
    }
    navigation = st.navigation(pages, position="sidebar")
    navigation.run()


if __name__ == "__main__":
    main()
