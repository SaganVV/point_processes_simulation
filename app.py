import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from bdm import BirthDeathMigration, HistoryTracker, ConfigEvaluator, BDM_states
from helper import acf_plot, cummean, mcmc_mean_variance_estimator
from point_process import StraussDensity, SaturatedDensity, PoissonDensity, HardcoreDensity


def merge_figure_traces(main_fig, fig):
    for trace in fig["data"]:
        main_fig.add_trace(trace)

class SimulationApp:
    session_states_names = ["history_pd", "configs"]
    def __init__(self):
        st.set_page_config(layout="wide")
        st.title("Point Process Simulation")
        for state in self.session_states_names:
            if state not in st.session_state:
                st.session_state[state] = None

    def get_user_inputs(self):
        st.sidebar.subheader("Density Parameters")
        density_select = st.sidebar.selectbox("Select point process", ("Poisson", "Hardcore", "Strauss", "Saturated Density"))

        beta = st.sidebar.number_input("Beta", 0.0, None, 10.0)

        if density_select == "Poisson":

            density = PoissonDensity(beta)
        if density_select == "Hardcore":
            R = st.sidebar.number_input("R", 0.0, None, 2.0)
            density = HardcoreDensity(beta, R)
        if density_select == "Strauss":
            R = st.sidebar.number_input("R", 0.0, None, 2.0)
            gamma = st.sidebar.number_input("Gamma", 0.0, 1.0, 0.99)
            density = StraussDensity(R, beta, gamma)

        if density_select == "Saturated Density":
            R = st.sidebar.number_input("R", 0.0, None, 2.0)
            gamma = st.sidebar.number_input("Gamma", 0.0, None, 1.5)
            saturation = st.sidebar.number_input("Saturation", 0, None, 0)
            density = SaturatedDensity(R, beta, gamma, saturation)

        params = {"density": density}
        params = {**params, **self.get_simulation_parameters()}
        return params

    def _get_normalized_probs(self):
        probs_columns = st.sidebar.columns(len(BDM_states))
        probs = [None] * len(BDM_states)
        for i, state in enumerate(BDM_states):
            with probs_columns[i]:
                probs[i] = st.number_input(f"{state.name}", 0.0, 1.0, 1 / len(BDM_states))
        normalized_probs = np.array(probs) / np.sum(probs)
        return normalized_probs

    def get_simulation_parameters(self):
        st.sidebar.subheader("Simulation Parameters")
        num_iterations = st.sidebar.number_input("Iterations", 0, max_value=None, value=2000)
        warm_up = st.sidebar.number_input("Warm Up iterations", 0, max_value=None, value=0)
        normalized_probs = self._get_normalized_probs()
        return {"num_iterations": num_iterations, "warm_up": warm_up, "probs": normalized_probs}

    def _run_pure_simulation(self, progress_callback=None, **params):
        tracker = HistoryTracker()
        num_of_points = ConfigEvaluator(function=lambda config: len(config))

        bdm = BirthDeathMigration(params["density"], state_sampler_probs=params["probs"])
        total_iter = params["num_iterations"] + params["warm_up"]
        for i, state in enumerate(bdm.run(num_iter=params["num_iterations"], warm_up=params["warm_up"], callbacks=[tracker, num_of_points])):
            if progress_callback and i % 100 == 0:
                progress_callback(i, total_iter)
        return tracker, num_of_points

    def _update_states(self, tracker, num_of_points):
        st.session_state.history_pd = pd.DataFrame({
            "iter": range(len(tracker.states)),
            "states": [state.name for state in tracker.states],
            "num_of_points": num_of_points.values,
            "acceptance": tracker.acc_probs,
            "is_accepted": tracker.were_accepted
        })
        st.session_state.configs = tracker.configs

    def run_simulation(self, **params):

        progress_bar = st.progress(0)
        running_sim_text = st.empty()
        running_sim_text.text("Running simulation")
        def progress_callback(current_iteration, total_iterations):
            progress_bar.progress(current_iteration / total_iterations)

        tracker, num_of_points = self._run_pure_simulation(progress_callback, **params)

        progress_bar.progress(1)
        progress_bar.empty()
        running_sim_text.text("Completed simulation").empty()
        self._update_states(tracker, num_of_points)

    def _render_configuration_plot(self):
        configs = st.session_state.configs
        config_container, config_num_col = st.columns([3, 2])
        config_num = config_num_col.number_input(
            label="Iteration", min_value=0, max_value=len(configs) - 1, value=len(configs) - 1, key="config_number"
        )
        config_num_col.write(f"Number of points: {len(configs[config_num])}")
        config_df = pd.DataFrame(configs[config_num], columns=["x", "y"])
        config_num_col.dataframe(config_df)
        fig_config = px.scatter(
            config_df,
            x="x",
            y="y",
            title=f"Configuration at iteration {config_num}",
            width=500,
            height=500,
            range_x=[0, 1],
            range_y=[0, 1],
        )

        config_container.plotly_chart(fig_config, use_container_width=False)

    def _render_point_statistics(self):
        history_pd = st.session_state.history_pd
        mean_var = mcmc_mean_variance_estimator(history_pd["num_of_points"])
        mean_std = np.sqrt(mean_var)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sample mean of number of points", f"{np.mean(history_pd['num_of_points']):.2f}")
        with col2:
            st.metric("Estimated Std. Error (MCMC)", f"{mean_std:.2f}")

        fig_points = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Number of points over iterations", "Distribution of number of points")
        )

        fig_points.add_trace(go.Scatter(x=history_pd["iter"], y=history_pd["num_of_points"],
                                        mode="lines", name="Number of points"), row=1, col=1)
        fig_points.add_trace(go.Scatter(x=history_pd["iter"], y=cummean(history_pd["num_of_points"]),
                                        mode="lines", name="Cumulative mean"), row=1, col=1)

        for trace in px.histogram(history_pd, x="num_of_points")["data"]:
            fig_points.add_trace(trace, row=1, col=2)

        st.plotly_chart(fig_points, use_container_width=True)

        fig_acf = acf_plot(history_pd["num_of_points"], nlags=len(history_pd)//20)

        st.plotly_chart(fig_acf, use_container_width=True)

    def _render_acceptance_analysis(self):
        history_pd = st.session_state.history_pd
        fig_accept = go.Figure(layout_title="Acceptance probabilities by proposal type", layout_yaxis_range=[0,1], layout_xaxis_range=[0, len(history_pd)])

        color_discrete_map = {'BIRTH': 'green', 'DEATH': 'red', 'MIGRATION': 'blue'}

        merge_figure_traces(fig_accept, px.scatter(history_pd, x="iter", y="acceptance",
                                                   color="states", color_discrete_map=color_discrete_map, opacity=0.15).update_traces(visible='legendonly'))

        for state in BDM_states:
            state_pd = history_pd[history_pd["states"] == state.name]
            fig_accept.add_trace(
                go.Scatter(
                    x=state_pd["iter"],
                    y=cummean(state_pd["acceptance"]),
                    line=dict(color=color_discrete_map[state.name]),
                    name=f"{state.name} (cumulative mean)",
                    mode="lines"
                )
            )
        st.plotly_chart(fig_accept, use_container_width=True)

    def run(self):
        inputs = self.get_user_inputs()
        tab1, tab2, tab3 = st.tabs(["Points", "Number of points statistics", "Acceptance Analysis"])

        if st.sidebar.button("Run Simulation"):
            self.run_simulation(**inputs)

        if st.sidebar.button("Clear"):
            for state in self.session_states_names:
                st.session_state[state] = None

        if st.session_state.configs is not None:
            with tab1:
                self._render_configuration_plot()
        if st.session_state.history_pd is not None:

            with tab2:
                self._render_point_statistics()
            with tab3:
                self._render_acceptance_analysis()


if __name__ == "__main__":
    app = SimulationApp()
    app.run()