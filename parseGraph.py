from rdflib import Namespace, RDF, RDFS, Graph

#memanggil file RDF
g = Graph()
g.parse("files/maizePestandDisease.rdf")

#Mendefinisikan namespaces
EX = Namespace("http://example.org/maize#")

#membuat graf RDF baru
g_new = Graph()
g_new.bind("ex", EX) #menambahkan namespace yang sudah dideklarasikan

#iterasi melalui semua triplets dalam graf hasil
for subj, pred, obj in g:
    g_new.add((subj, pred, obj))

#definisikan ontologi (kelas dan properti)
g_new.add((EX.Tanaman, RDF.type, RDFS.Class))
g_new.add((EX.Hama, RDF.type, RDFS.Class))
g_new.add((EX.Penyakit, RDF.type, RDFS.Class))
g_new.add((EX.Gejala, RDF.type, RDFS.Class))
g_new.add((EX.Lingkungan, RDF.type, RDFS.Class))

g_new.add((EX.menyerang, RDF.type, RDF.Property))
g_new.add((EX.menyebabkan, RDF.type, RDF.Property))
g_new.add((EX.memilikiGejala, RDF.type, RDF.Property))
g_new.add((EX.disebabkanOleh, RDF.type, RDF.Property))

#ekspor agar file tersebut berada di tempat yang sudah ditentukan
g_new.serialize(destination="files/p_and_d_new.rdf", format="xml")