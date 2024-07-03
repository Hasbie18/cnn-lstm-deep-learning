from rdflib import Graph, Namespace
import pandas as pd

g = Graph()
g.parse("files/maizePestandDisease.rdf", format='xml')


EX = Namespace("http://example.org/maize#")

#ekstrak informasi dari file RDF tersebut

data = []
for subj, pred, obj in g:
    if pred == EX.name:
        name = obj
    elif pred == EX.menyerang:
        tanaman = obj
    elif pred == EX.menyebakan:
        penyakit = obj
    elif pred == EX.memilikiGejala:
        gejala = obj
    else:
        continue
    data.append({
        "Subject" : subj,
        "Predicate" : pred,
        "Object" : obj
    })

#membuat data frame yang dieksrak
df = pd.DataFrame(data)

#menyusun data ke dalam kolom yang sesuai
data_list = []
for s, p, o in g.triples((None, None, None)):
    hama_name = g.value(s, EX.name)
    tanaman = g.value(s, EX.menyerang)
    penyakit = g.value(s, EX.menyebabkan)
    gejala = g.value(s, EX.memilikiGejala) if penyakit else None

    data_list.append({
        "Hama" : hama_name,
        "Tanaman" : tanaman,
        "Penyakit" : penyakit,
        "Gejala" : gejala,
    })

df = pd.DataFrame(data_list)

df.to_csv("files/m_and_p_data.csv", index=False)
print("Berhasil menyimpan")