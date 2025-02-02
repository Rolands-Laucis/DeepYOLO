.\env.ps1
# python lstm2.py --full=True --csvs_per_run=4000 --rnn="gru" --units=64 --epochs=10
python lstm2.py --full=True --csvs_per_run=4000 --rnn="gru" --units=128 --epochs=10
# python lstm2.py --full=True --csvs_per_run=4000 --rnn="gru" --units=192 --epochs=10
# python lstm2.py --full=True --csvs_per_run=800 --rnn="lstm"
# python lstm2.py --model="models\\LSTM_loss4165937920.0_a0.829_rmse64542.993_u64_e50_d21_w8_m6_c3.json" --full=True --csvs_per_run=100