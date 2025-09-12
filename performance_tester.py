from point_process import StraussDensity, PoissonDensity
from bdm import BirthDeathMigration, HistoryTracker, ConfigEvaluator

from cProfile import Profile
from pstats import SortKey, Stats
import datetime

test_folder = 'speed_tests\\'
if __name__ == "__main__":
    R = 0.1
    beta = 100
    gamma = 0.5
    strauss_density = PoissonDensity(beta)#StraussDensity(R, beta, gamma)
    print(strauss_density)
    num_iter = 20000
    bdm = BirthDeathMigration(strauss_density)

    tracker = HistoryTracker()
    num_of_points = ConfigEvaluator(function=lambda config: len(config))
    filename = test_folder + f"func=bdm.run;num_iter={num_iter};{strauss_density!r};{datetime.datetime.now().strftime("%d-%m-%Y;%H-%M-%S")}.txt"
    with Profile() as profile:
        for i, state in enumerate(
            bdm.run(num_iter=num_iter, callbacks=[tracker, num_of_points])
        ):
            pass
        with open(filename, 'w') as file:
            (Stats(profile, stream=file).strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats())
