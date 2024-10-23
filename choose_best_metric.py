from pathlib import Path
import re
import numpy as np

# metrics_path = Path('output/exp_17/metrics.txt')
metrics_path = Path('output_pol/exp_42/metrics.txt')

metrics_criteria = 'mean iou:'
# metrics_criteria = 'mean iou (strict):'

epoch_num = 1
metric_values = []
with open(metrics_path) as f:
    for line in f:
        if metrics_criteria in line:
            metric_value = re.findall("\d+\.\d+", line)[0]
            metric_values.append(metric_value)
            print(f'Metric value {metric_value} for epoch {epoch_num}')
            epoch_num += 1

print(metric_values)

index = np.argmax(metric_values)
print(f'Best metric value is {metric_values[index]} for epoch {index+1} for {metrics_path}')
