from rdflib import Graph


g = Graph()
g.parse("files/maizePestandDisease.rdf", format='xml')

#untuk memeriksa jumlah triplets dari file maize and pest disease
print(f"total triplets : {len(g)}")

for subj, pred, obj, in g:
    print(f"subject: {subj} \n predicate: {pred} \n object: {obj}")
    print("-----------------------------------")