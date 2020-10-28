import pickle        
import base64
import csv

your_pickle_obj = pickle.loads(open('results/Dist-TMY3.pkl', 'rb').read())
with open('output.csv', 'a', encoding='utf8') as csv_file:
    wr = csv.writer(csv_file, delimiter='|')
    pickle_bytes = pickle.dumps(your_pickle_obj)            # unsafe to write
    b64_bytes = base64.b64encode(pickle_bytes)  # safe to write but still bytes
    b64_str = b64_bytes.decode('utf8')          # safe and in utf8
    wr.writerow(['col1', 'col2', b64_str])

