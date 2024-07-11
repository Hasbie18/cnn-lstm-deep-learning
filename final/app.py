import re
import time
from rdflib import Graph,Namespace,URIRef,Literal
from pyfiglet import Figlet 
from colorama import init, Fore, Style, Back

init()

g = Graph()
EX = Namespace("http://www.semanticweb.org/user/ontologies/2023/11/untitled-ontology-40#")
RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
g.bind("rdfs", RDFS)
g.bind("ex", EX)
g.parse("data/dataku.rdf", format='xml')
b = "http://www.semanticweb.org/user/ontologies/2023/11/untitled-ontology-40#"

#! Buat Inputan User
def prepocessingData(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text

def loadGejala():
    list_gejala = []
    for s,p,o in g:
            if p == EX.memilikiGejala:
                list_gejala.append(get_object_name(o))
    return list_gejala

def deteksiGejala(processed_input, processed_target_words):
    found_words = []
    for word in processed_target_words:
        if word.replace("_", " ") in processed_input:
            found_words.append(word)
    return found_words

def cariGejala(gejala):
    result = []
    for s,p,o in g:
        if p == EX.memilikiGejala:
            if gejala in str(o):
                result.append(s)
    return result

#! OUTPUT
def cariIndividual(indiv):
    properties = {}
    for s, p, o in g.triples((indiv, None, None)):
        if isinstance(o, (str, int, float)):
            properties[p] =o
        else:
            properties[p] = o
    return properties



def get_object_name(obj):
    if isinstance(obj, URIRef):
        return obj.split("#")[-1]  # Mengambil bagian akhir URI setelah karakter '#'
    elif isinstance(obj, Literal):
        return obj.toPython()  # Mengambil nilai teks dari Literal
    return None

def get_object_value(subjek):
    # Mencari nilai objek yang terkait dengan subjek dan properti tertentu
    nilai_objek = None
    for s, p, o in g.triples((subjek, RDFS['abstract'], None)):
        nilai_objek = get_object_name(o)
        break  # Mengambil hanya nilai pertama yang ditemukan
    
    return nilai_objek

def print_separator():
    print(Fore.GREEN + '============================================================')


def main():
    # d = Figlet(font='slant')
    f = Figlet(font='slant',width=200)
    print(Fore.YELLOW + f.renderText(' SISTEM DIAGNOSA ') + Style.RESET_ALL)
    count = True
    while count:
        x = str(input(f"{Fore.LIGHTWHITE_EX }Masukkan gejala yang dicari (ketik 'exit' untuk keluar): "))
        if x.lower() == 'exit':
            count = False
            continue

        list_gejala = []
        for i in loadGejala():    
            text = i.lower()
            text = re.sub(r'\W+', ' ', text)
            text = re.sub(r'\d+', '', text)
            list_gejala.append(text)

        inputan = x.lower()
        inputan = re.sub(r'\W+', ' ', inputan)
        inputan = re.sub(r'\d+', '', inputan)

        deteksi_gejala = deteksiGejala(inputan, list_gejala)

        print_separator()
        print(f"Analisis Kata: {x}")
        print_separator()
        time.sleep(1)

        if deteksi_gejala:
            print(Fore.YELLOW + "Gejala Terdeteksi:")
            for z in deteksi_gejala:
                time.sleep(0.5)
                print(Fore.YELLOW + f"  - {z.replace('_', ' ')}")
        else:
            print(Fore.YELLOW + "Tidak ada gejala yang terdeteksi.")

        print_separator()

        for z in deteksi_gejala:
            time.sleep(1)
            gejala = cariGejala(z)
            if gejala:
                for i in gejala:
                    time.sleep(1)
                    properties = cariIndividual(i)
                    if properties:
                        name = get_object_name(i)
                        print(Fore.CYAN + f"\n {z.replace('_', ' ')} disebabkan oleh {name.replace('_',' ')}:")
                        print_separator()
                        time.sleep(1)
                        for prop, value in properties.items():
                            fitur = get_object_name(prop)
                            hasill = get_object_name(value)
                            abstract = get_object_value(value)
                            hasil = hasill.replace("_", " ")
                            time.sleep(1)
                            if abstract:
                                print(f"{Fore.CYAN + str(fitur)}: {Fore.YELLOW + str(hasil)}")
                                print(Fore.WHITE + f"{str(abstract)}")
                                print()
                            elif fitur == 'memilikiGejala':
                                print()
                            else:
                                print(f"{Fore.CYAN + str(fitur)}: {Fore.YELLOW + str(hasil)}")
                                print()
                        print_separator()
                    else:
                        print(Fore.CYAN + f"Tidak ditemukan properti dari individu {i}.")
            else:
                print(Fore.CYAN + 'Tidak dapat menemukan gejala')

    print(Fore.YELLOW + f.renderText('TERIMA KASIH') + Style.RESET_ALL)
    print(Fore.YELLOW + "Terima kasih telah menggunakan SITEM DIAGNOSIS." + Style.RESET_ALL)


if __name__ == "__main__":
    main()
