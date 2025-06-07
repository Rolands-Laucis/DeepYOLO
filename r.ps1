.\env.ps1
# python model.py
# python lstm.py
# python lstm2.py #--full=True
# python lstm2.py --csvs=400 --csvs_per_run=400 --epochs=20 --rnn="lstm" --units=128 --save=0 --full=True
# python lstm2.py --csvs=100 --csvs_per_run=100 --epochs=1 --rnn="lstm" --units=64 --save=0 --full=True

# python lstm3.py --csvs_per_run=1 --epochs=1 --rnn="lstm" --units=64 --save=0 --full=True
python lstm3.py --csvs=1 --csvs_per_run=1 --epochs=1 --rnn="lstm" --units=64 --save=0

# python tft.py
# python tft2.py