.\env.ps1
# python lstm2.py --full=True --csvs_per_run=4000 --rnn="gru" --units=64 --epochs=10
# python lstm2.py --full=True --csvs_per_run=5000 --rnn="gru" --units=128 --epochs=20
# python lstm2.py --full=True --csvs=100 --csvs_per_run=1000 --rnn="lstm" --units=128 --epochs=20 --norm
# python lstm2.py --full=True --csvs_per_run=4000 --rnn="gru" --units=192 --epochs=10
# python lstm2.py --full=True --csvs_per_run=1200 --rnn="lstm" --units=64 --epochs=10 --model="lstm_loss49.162_a0.727_rmse7.011_u64_e10_csvs9146_d21_w8_c5.json"
# python lstm2.py --model="models\\LSTM_loss4165937920.0_a0.829_rmse64542.993_u64_e50_d21_w8_m6_c3.json" --full=True --csvs_per_run=100
python lstm3.py --full=True --csvs_per_run=1200 --rnn="lstm" --units=64 --epochs=10

# python tft.py