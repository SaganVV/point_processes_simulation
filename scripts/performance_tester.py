from src.densities import PoissonDensity
from src.samplers.bdm import BirthDeathMigration, HistoryTracker, ConfigEvaluator

from cProfile import Profile
from pstats import SortKey, Stats
import datetime

test_folder = "performance_results\\"
if __name__ == "__main__":
    R = 0.1
    beta = 100
    gamma = 0.5
    gamma_saturated = 1.2
    saturation = 3
    random_seed = 42
    strauss_density = PoissonDensity(beta=beta)
    # strauss_density = SaturatedDensity(R, beta, gamma=gamma_saturated, saturation=saturation)
    # strauss_density = StraussDensity(R, beta, gamma)
    print(strauss_density)
    num_iter = 40000
    bdm = BirthDeathMigration(strauss_density, seed=random_seed)

    tracker = HistoryTracker()
    num_of_points = ConfigEvaluator(function=lambda config: len(config))
    filename = (
        test_folder
        + f"{datetime.datetime.now().strftime("%d-%m-%Y;%H-%M-%S")};func=bdm.run;num_iter={num_iter};{strauss_density!r};.txt"
    )
    print(filename)
    with Profile() as profile:
        for i, state in enumerate(
            bdm.run(num_iter=num_iter, callbacks=[tracker, num_of_points])
        ):
            pass

        with open(filename, "w") as file:
            (
                Stats(profile, stream=file)
                .strip_dirs()
                .sort_stats(SortKey.TIME)
                .print_stats()
            )
