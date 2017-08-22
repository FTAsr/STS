## Contains the configurable parameters for the Short Answer Scoring system.

## Set the required grading scale as True
GRADING_SCALE_MULTICLASS = False
GRADING_SCALE_REAL_0_5 = True
GRADING_SCALE_REAL_0_1 = False
GRADING_SCALE_Integer_0_5 = False
GRADING_SCALE_LABELS = ['Incorrect', 'Partially Correct', 'Correct']

PARTIALLY_CORRECT_LOW = 0
PARTIALLY_CORRECT_HIGH = 3

## Set the required mode of operation
INTERACTIVE_MODE = True
BATCH_MODE = False

## Threshold for feedback
LOWER_LIMIT = 15.0

## Set the pre-trained classifier
BEST_CLASSIFIER_COLLEGE = 'feed+fb+college.file'
BEST_CLASSIFIER_1A = 'bow+fb+1A.file'
BEST_CLASSIFIER_2A = 'bow+fb+2A.file'
BEST_CLASSIFIER_SICK = 'bow+fb+sick.file'
BEST_CLASSIFIER_STS = 'bow+fb+sts.file'

# Output folder for Batch mode testing
OUTPUT_PATH = ''
