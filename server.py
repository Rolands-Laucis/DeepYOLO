from infer import Model, API
import pathlib
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Process some strings.')
parser.add_argument('ticker', type=str, nargs='?', default='RDDT', help='An optional string input argument')
args = parser.parse_args()

if False:
    import json
    with open('resp.json', 'r') as j:
        d = json.load(j)
else:
    api = API()
    d = api.RequestDaily(sym=args.ticker)
    if False:
        print(f'{args.ticker} last day OHLC:')
        # print(['Open', 'High', 'Low', 'Close'])
        print(d[list(d.keys())[0]])

models_base_path = pathlib.Path('./models')
m = Model()
models = {
    # 'GRU': models_base_path / 'gru_loss122.637_a0.723_rmse11.074_u128_e20_csvs11466_d21_w8_c5.json',
    # 'GRU yesterday': models_base_path / 'gru_loss43.736_a0.674_rmse6.613_u128_e10_csvs4000_d21_w8_c5.json',
    # 'LSTM best': models_base_path / 'lstm_loss62.973_a0.714_rmse7.936_u128_e20_csvs11466_d21_w8_c5.json',
    # 'LSTM 3 layers': models_base_path / 'lstm_loss30.498_a0.816_rmse5.522_u64_e10_csvs9146_d21_w8_c5.json',
    # 'LSTM BIG 4 layers 50D 3C': models_base_path / 'lstm_loss20.19_a1.0_rmse4.493_u90_e4_csvs9147_d50_c3.json',
    # 'LSTM BIG 4 layers 30D 5C 39L': models_base_path / 'lstm_loss39.878_a1.0_rmse6.305_u128_e5_csvs9147_d30_c5.json',
    'LSTM BIG 4 layers 30D 5C 47L': models_base_path / 'lstm_loss47.664_a0.996_rmse6.891_u128_e5_csvs9147_d30_c5.json',
    # 'LSTM BIG 4 layers 30D 5C 57L': models_base_path / 'lstm_loss57.045_a1.0_rmse7.541_u128_e5_csvs9147_d30_c5.json',
}

candle_df = Model.ParseAPIData(d)
last_days = 2
print(f'{args.ticker} last {last_days} day OHLC:')
print(candle_df.head(last_days))

for name, model_json in models.items():
    m.LoadModel(model_json, name)

    x = m.PrepDataDays(d)
    pred = m.pred(x, ticker=args.ticker)

    print(pred.head())

# x[0] = x[0][:,:,1:]
# x[1] = x[1][:,:,1:]
# m.LoadModel(models_base_path / 'lstm_loss24.793_a0.663_rmse4.979_u128_e20_csvs11466_d21_w8_c5.json', 'LSTM without date')
# print(m.pred(x))