from point_process import StraussDensity
from bdm import BirthDeathMigration, HistoryTracker, ConfigEvaluator

from cProfile import Profile
from pstats import SortKey, Stats

if __name__ == "__main__":
    R = 0.5
    beta = 100
    gamma = 0.9
    strauss_density = StraussDensity(R, beta, gamma)
    bdm = BirthDeathMigration(strauss_density)

    tracker = HistoryTracker()
    num_of_points = ConfigEvaluator(function=lambda config: len(config))
    with Profile() as profile:
        for (i, state) in enumerate(bdm.run(num_iter=10000, callbacks=[tracker, num_of_points])):
            pass
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.CALLS)
            .print_stats()
        )


    # history_pd = pd.DataFrame({"iter": range(len(tracker.states)), "states": tracker.states, "num_of_points": num_of_points.values, "acceptance": tracker.acc_probs, "is_accepted": tracker.were_accepted})
    # print(history_pd.head(100))
    # fig, axs = plt.subplots(2, 2)
    # sns.lineplot(history_pd["num_of_points"], ax=axs[0, 0])
    # sns.lineplot(cummean(history_pd.num_of_points), ax=axs[0, 0])
    # sns.histplot(history_pd["num_of_points"], ax=axs[0, 1])
    # sns.scatterplot(history_pd, x="iter", y="acceptance", hue=history_pd["states"], ax=axs[1, 0])
    # for state in [BDM_states.BIRTH, BDM_states.DEATH, BDM_states.MIGRATION]:
    #     state_hist = history_pd[history_pd["states"] == state]
    #     sns.lineplot(cummean(state_hist.acceptance), ax=axs[1,1], label=str(state))
    # sns.lineplot(cummean(history_pd.acceptance), ax=axs[1,1], label="overall")
    # plt.tight_layout()
    # plt.show()

