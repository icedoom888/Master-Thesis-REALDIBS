import sys
import os
from viztool.viztool_crop_and_evaluate import crop_and_evaluate
from viztool.viztool_evaluate_with_csv import evaluate_with_csv

# HOW TO:

### REQUIREMENTS ###
# 1) Run viztool_pygame
# 2) Save a few clicks
# 3) provide BASE_PATH which is the base folder of the experiments, same path required by viztool_pygame

# This script will generate folders for each click, computing metrics, found in BASE_FOLDER_crops
# Metrics will be saved for display purposes in BASE_FOLDER_metrics

data_path = sys.argv[1]
crop_path = sys.argv[1][:-1] + '_viztool_screenshots/'
output_path = sys.argv[1][:-1] + '_crops/'

if os.path.isdir(crop_path):
    crop_and_evaluate(data_path, crop_path, output_path)

else:
    print('\nYou have to run viztool_pygame on %s first.' % data_path)

data_path = output_path
output_path = sys.argv[1][:-1] + '_metrics/'

evaluate_with_csv(data_path, crop_path, output_path, 'comparison_with_image')
evaluate_with_csv(data_path, crop_path, output_path, 'comparison_with_difference')
evaluate_with_csv(data_path, crop_path, output_path, 'comparison_transpose')
