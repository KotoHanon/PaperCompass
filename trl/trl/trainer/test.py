import sys
import os

current_file = os.path.abspath(__file__)
dir_trainer = os.path.dirname(current_file)
dir_main = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
dir_vllm = os.path.join(dir_main, 'vllm/vllm')
print(dir_vllm)