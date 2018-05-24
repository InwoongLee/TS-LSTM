#!/bin/sh

python3 Generalized_TS-LSTM.py 1 2 3 4 &&
python3 Generalized_TS-LSTM.py 1 3 2 4 &&
python3 Generalized_TS-LSTM.py 1 4 2 3 &&
python3 Generalized_TS-LSTM.py 2 3 1 4 &&
python3 Generalized_TS-LSTM.py 2 4 1 3 &&
python3 Generalized_TS-LSTM.py 3 4 1 2