from infer import Model, API, PrepData

if False:
    import json
    with open('resp.json', 'r') as j:
        d = json.load(j)
        x = PrepData(d)
        print(x[0][0][0][1:5])
        exit(0)

ticker = 'RDDT'
api = API()
d = api.RequestDaily(sym=ticker)
x = PrepData(d)
if False:
    print(f'{ticker} last day OHLC:')
    print(['Open', 'High', 'Low', 'Close'])
    print(x[0][0][0][1:5])

m = Model()
m.LoadModel('models\\gru_loss122.637_a0.723_rmse11.074_u128_e20_csvs11466_d21_w8_c5.json', 'GRU')
print(m.pred(x))

m.LoadModel('models\\gru_loss43.736_a0.674_rmse6.613_u128_e10_csvs4000_d21_w8_c5.json', 'GRU yesterday')
print(m.pred(x))

m.LoadModel('models\\lstm_loss62.973_a0.714_rmse7.936_u128_e20_csvs11466_d21_w8_c5.json', 'LSTM')
print(m.pred(x))