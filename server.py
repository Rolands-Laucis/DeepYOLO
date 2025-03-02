from infer import Model, API
import pathlib
import argparse

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
    if True:
        print(f'{args.ticker} last day OHLC:')
        # print(['Open', 'High', 'Low', 'Close'])
        print(d[list(d.keys())[0]])

models_base_path = pathlib.Path('./models')
m = Model()
models = {
    # 'GRU': models_base_path / 'gru_loss122.637_a0.723_rmse11.074_u128_e20_csvs11466_d21_w8_c5.json',
    # 'GRU yesterday': models_base_path / 'gru_loss43.736_a0.674_rmse6.613_u128_e10_csvs4000_d21_w8_c5.json',
    # 'LSTM best': models_base_path / 'lstm_loss62.973_a0.714_rmse7.936_u128_e20_csvs11466_d21_w8_c5.json',
    'LSTM 3 layers': models_base_path / 'lstm_loss30.498_a0.816_rmse5.522_u64_e10_csvs9146_d21_w8_c5.json',
}

for name, model_json in models.items():
    m.LoadModel(model_json, name)
    x = m.PrepData(d)
    print(m.pred(x, ticker=args.ticker))

# x[0] = x[0][:,:,1:]
# x[1] = x[1][:,:,1:]
# m.LoadModel(models_base_path / 'lstm_loss24.793_a0.663_rmse4.979_u128_e20_csvs11466_d21_w8_c5.json', 'LSTM without date')
# print(m.pred(x))