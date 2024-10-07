from data_gen import generate_floor_plan, genetic_algorithm

import numpy as np
from PIL import Image
import datetime
import time

# define greyscale mapping
GRAYSCALE_MAPPING = np.array([
    0, #Black
    255 #White
], dtype=np.uint8)

# define coloured mapping
COLOR_MAPPING = np.array([
    [255, 0, 0],      # Red for -1
    [255, 255, 255],  # White for 0
    [0, 0, 255]       # Blue for 1
], dtype=np.uint8)


def re_scale(img,scale_factor=50):
    return img.resize((img.width * scale_factor, img.height * scale_factor), Image.NEAREST)

# Run the genetic algorithm
def generate_sample():
    floor_plan = generate_floor_plan()
    best_solution = genetic_algorithm(floor_plan)

    img_solution = Image.fromarray(GRAYSCALE_MAPPING[best_solution])
    img_floor = Image.fromarray(GRAYSCALE_MAPPING[floor_plan])
    #img_added = Image.fromarray(COLOR_MAPPING[floor_plan-best_solution+1])

    return img_solution, img_floor


def generate_data(sample_count=1):
    for i in range(sample_count):
        x,y = generate_sample()
        x.save(f'data/x/{i+1}.png')
        y.save(f'data/y/{i+1}.png')

def test_generation(sample_count=1):
    st = time.monotonic()
    x,y = generate_sample()
    x.save(f'data/x/0.png')
    y.save(f'data/y/0.png')
    et = time.monotonic()
    duration = et-st
    print(duration)
    return duration


def time_to_minutes(seconds):
    return datetime.timedelta(seconds=seconds)



if __name__ == "__main__":
    sample_count = 2000
    duration = test_generation()
    duration = time_to_minutes(duration*sample_count)
    print(duration)
    generate_data(sample_count=sample_count)

