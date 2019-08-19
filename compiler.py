data = open('data/MAERSK-B.csv')

data = data.read().split(';')

new_data = []

for string in data:
    new_string = string
    for i, c in enumerate(string):
        if c == ',':
            new_string = string[:i] + '.' + string[i+1:]
    new_data.append(new_string)

new_data = ",".join(new_data)

with open('data/MAERSK_DATA.csv', 'w') as wf:
    wf.write(new_data)