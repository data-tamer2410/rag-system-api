import requests
import time
from statistics import mean

n = 25
input = {'content': 'Tell me about a country that is famous for football.'}
all_times = []
url = 'http://127.0.0.1:8000/generate/'

for _ in range(n):
    start_time = time.time()  
    response = requests.post(url, json=input)
    if response.status_code == 200:
        end_time = time.time()  
        res = end_time - start_time
        all_times.append(res)
    else:
        print(f'Error: {response.status_code}')  

print(f'Max time: {max(all_times)}')
print(f'Min time: {min(all_times)}')
print(f'Mean time: {mean(all_times)}')
