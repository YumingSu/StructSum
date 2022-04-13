from openie import StanfordOpenIE
from models.modules.BiLSTMEncoder import BiLSTMEncoder

properties = {
    'openie.affinity_probability_cap': 2 / 3,
}

with StanfordOpenIE(properties=properties) as client:
    # text = 'Barack Obama was born in Hawaii. Richard Manning wrote this sentence.'
    text = 'At the beginning of 2020, an ice tsunami occurred along Lake Erie in the United States. The floating ice formed an "ice wall" more than 9 meters high under the waves, destroying buildings and other facilities around Lake Erie; in November 2019, the ice tsunami "occurred" in Russia Found off the ground, residents evacuated'
    print('Text: %s.' % text)
    for triple in client.annotate(text):
        print('|-', triple)

    with open('corpus/pg6130.txt', encoding='utf8') as r:
        corpus = r.read().replace('\n', ' ').replace('\r', '')

    triples_corpus = client.annotate(corpus[0:5000])
    print('Corpus: %s [...].' % corpus[0:80])
    print('Found %s triples in the corpus.' % len(triples_corpus))
    for triple in triples_corpus[:3]:
        print('|-', triple)
    print('[...]')
