list = [{'No': '1A', 'item': 'XP30000'}, {'No': '2A', 'item': 'トレジャーレーダー'}, {'No': '3A', 'item': 'XP10000'}, ]

g = []
for i in list:
    if 'item' in i:
        result = i['item']
        g.append(result)
        print(result)
print(g)