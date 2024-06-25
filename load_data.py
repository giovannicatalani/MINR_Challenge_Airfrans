import os
from lips import get_root_path
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark
from lips.dataset.airfransDataSet import download_data
from my_augmented_simulator import AugmentedSimulator
from lips.evaluation.airfrans_evaluation import AirfRANSEvaluation

# indicate required paths
LIPS_PATH = get_root_path()
ROOT_PATH      = "/home/dmsm/gi.catalani/Projects/Challenge_Airfrans/"
DIRECTORY_NAME = '/scratch/dmsm/gi.catalani/Airfrans/Dataset'
BENCHMARK_NAME = "Case1"
LOG_PATH = LIPS_PATH + "lips_logs.log"

BENCH_CONFIG_PATH = os.path.join(ROOT_PATH,"submissions","airfoilConfigurations","benchmarks","confAirfoil.ini") #Configuration file related to the benchmar


if not os.path.isdir(DIRECTORY_NAME):
    download_data(root_path=".", directory_name=DIRECTORY_NAME)

benchmark=AirfRANSBenchmark(benchmark_path = DIRECTORY_NAME,
                            config_path = BENCH_CONFIG_PATH,
                            benchmark_name = BENCHMARK_NAME,
                            log_path = LOG_PATH)
benchmark.load(path=DIRECTORY_NAME)


sim = AugmentedSimulator(benchmark)
sim.train(benchmark.train_dataset)

predictions, observations = sim.predict(benchmark._test_dataset)
#predictions_ood, observations_ood = sim.predict(benchmark._test_ood_dataset)

evaluator = AirfRANSEvaluation(config_path = BENCH_CONFIG_PATH,
                               scenario = BENCHMARK_NAME,
                               data_path = DIRECTORY_NAME,
                               log_path = LOG_PATH)

observation_metadata = benchmark._test_dataset.extra_data
metrics = evaluator.evaluate(observations=observations,
                             predictions=predictions,
                             observation_metadata=observation_metadata)
print(metrics)

#observation_metadata = benchmark._test_ood_dataset.extra_data
#metrics = evaluator.evaluate(observations=observations_ood,
                            # predictions=predictions_ood,
                            # observation_metadata=observation_metadata)
#print(metrics)
#