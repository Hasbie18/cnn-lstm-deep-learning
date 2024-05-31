from rdflib import Graph

g = Graph()
g.parse("files/maizePestandDisease.rdf")

query = """ 
SELECT ?subject ?predicate ?object
WHERE {
  ?subject ?predicate ?object
}
"""

results = g.query(query)
for row in results:
  print(f"Subject : {row.subject}, Predicate : {row.predicate}, Object : {row.object}")