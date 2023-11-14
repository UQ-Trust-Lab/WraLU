import time

from kact.utils import get_bounds_of_variables
from kact.volume import InputConstraintsGenerator, SamplePointsGenerator, VolumeEstimator

if __name__ == '__main__':
    constrs_generator = InputConstraintsGenerator(4)
    constraints = constrs_generator.generate("box+random", -2, 2, 10)

    lower_bounds, upper_bounds = get_bounds_of_variables(constraints)
    print(lower_bounds, upper_bounds)

    sample_points_generator = SamplePointsGenerator(lower_bounds, upper_bounds)
    random_points = sample_points_generator.generate(1000000, "box", "relu")

    volume_estimator = VolumeEstimator(constraints, lower_bounds, upper_bounds)

    start = time.time()
    volume_estimator.estimate("grid")
    print("time: ",time.time() - start)

    start = time.time()
    volume_estimator.estimate("random", random_points)
    print("time: ",time.time() - start)
