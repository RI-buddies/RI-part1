from urllib.request import urlopen
from urllib import parse
from html.parser import HTMLParser
import requests
from bs4 import BeautifulSoup
import time

#Função para Salvar as páginas em arquivos
def salvaPagina(page,urlbase,visitadas,url):
    nome = 'htmls\\' + str(urlbase[12:])+ str(visitadas) + ".html"
    nome2 = 'htmls\\' + str(urlbase[12:])+ str(visitadas) + ".txt"
    print("\n salvando pagina com nome:\n",nome,"\n")

    arquivo = open(nome,'w')
    arquivo.write(str(page.encode('utf-8')))
    arquivo.close()

    arquivo2 = open(nome2,'w')
    arquivo2.write(url)
    arquivo2.close()

#Função para captura das páginas
def Pegalinks(url,visitadas,urlbase):

  response = requests.get(url)
  page = str(BeautifulSoup(response.content, 'html.parser'))
  #Salva o html da pagina visitada
  salvaPagina(page,urlbase,visitadas,url)

  linksRel = []
  linksRelGuitar = []
  soup = BeautifulSoup(page,'html.parser')
  #Identificação das tags ancoras para buscar links
  ancoras = soup.find_all('a')

  for elemento in ancoras:

    relev,isguitar = limpa(elemento)
    link = elemento.get('href')
    #Filtrando links invalidos
    if(relev != 0 and(link != None) and link != "" and link != "#" and link.find("javascript") == -1 and link.find(".aspx") == -1 and link.find("link") == -1 and link.find(" ") == -1):

        if(link[0]== '/' and link [1] =='/'):
            link = link[2:]
        #Adicionando urls nas lista
        if( (link.find('https://') == -1 and link.find('http://') == -1) and (link[0] != 'w' and link[3] != '.') ):
          if(link[0] == '/'):
            if(isguitar == 1):
              linksRelGuitar.append(urlbase + link)
            else:
              linksRel.append(urlbase + link)
          else:
            if(isguitar == 1):
              linksRelGuitar.append(urlbase + '/' + link)
            else:
              linksRel.append(urlbase + '/' + link)
        else:
          if(link[0] == 'w'):
              link = 'http://' + link
          if(isguitar == 1):
            linksRelGuitar.append(link)
          else:
            linksRel.append(link)

  return linksRelGuitar , linksRel
#filtro de palavras chaves (Lista1) e limpeza de links não desejaveis que são comuns(Lista2)
def limpa(elemento):
        lista1 = ["Viola","Violao","acustico","eletrico","guitar","nylon","aco","austic","Cordas","Instrumentos","Corda","instrumento","produto","product","violao-s2006"]
        lista2 = ["Twitter","Youtube","Instagram","Facebook","violino","hotsite","checkout","login","Login","signin"]
        isrele = 0
        isguitar = 0
        elemStr = str(elemento)
        for palavra in lista1:
            if (elemStr.find(palavra) != -1):
                isrele = isrele + 1
                break
        for palavra in lista2:
            if (elemStr.find(palavra) != -1):
                isrele = 0
                break
        if(isrele != 0):
          isguitar = relevanteGuitar(elemStr)

        return isrele , isguitar
#Após limpesa de palavras verifico a estrutura para definir relevancia da pagina , anilisado pelas paginas
def relevanteGuitar(elemStr):
  retorno = 0
  lista1 = ["violao-s2006/","product","/produto/","/dp/","artigo","/p/","/p?","?recsource","title-prod","cs-product","violao","Guitar","-p0","-p1","-p2","-p3","-p4","-p5","-p6","-p7","-p8","-p9"]
  lista2 = ["productCluster","product.catalog","/parceiros/","product-reviews"]
  for palavra in lista1:
    if(elemStr.find(palavra) != -1):
      retorno = 1
      break
  for palavra in lista2:
    if (elemStr.find(palavra) != -1):
      retorno = 0
      break

  return retorno
#verifica se a pagina já foi visitada
def verificaVis(visitadas , paraVisitar):
  j = 0
  retorno = ""
  outraPagina = 0
  for pagina in paraVisitar:
    for vis in visitadas:

      if(vis == pagina):
        outraPagina = 1
        break
    if(outraPagina == 1):
      j = j + 1
      outraPagina = 0
    else:
      retorno = paraVisitar[j]
      break

  return retorno , j+1

#Função principal, onde sera construida a arvore
def Crawler(url, maxPages, arquivo):
  PageTovisit = [url]
  PageTovisitGuitar = []
  visitadas = []
  numberVisited = 0
  logline = ""
  log = open(arquivo,'w')

  i = 3
  j = 0

  for char in url:
      if url[j] == '/' :
        i = i - 1
        if (i == 0):
          break
      j = j + 1

  urlbase = url[:(j - len(url))]
  #Loop de construção, é verificada se a pagina foi visitada, se não foi é colocada para visitação
  while numberVisited < maxPages and PageTovisit != []:
      numberVisited = numberVisited + 1
      url = ""
      url , j = verificaVis(visitadas,PageTovisitGuitar)
      PageTovisitGuitar = PageTovisitGuitar[j:]
      if(url == ""):
        url , j = verificaVis(visitadas,PageTovisit)
        PageTovisit = PageTovisit[j:]

      try:
          logline = (str(numberVisited)+ " " + "visiting: " + url + "\n")
          print(logline)
          log.write(logline)
          linksRelGuitar, linksRel = Pegalinks(url,numberVisited,urlbase)
          visitadas.append(url)
          PageTovisitGuitar = PageTovisitGuitar + linksRelGuitar
          PageTovisit = PageTovisit + linksRel
          #tempo entre requisições
          time.sleep(1)

      except Exception as ex:
          print(ex)
          print("ERRO")
          return 0

  log.close()
  return 1

Crawler("http://www.novamusic.com.br/",50,"Novamusic.txt")
Crawler("https://www.americanas.com.br/categoria/instrumentos-musicais",50,"Americanas.txt")
Crawler("https://www.multisom.com.br/",50,"Multisom.txt")
Crawler("https://www.mundomax.com.br/instrumentos-musicais",50,"Mundomax.txt")
Crawler("https://www.madeinbrazil.com.br",50,"MadeinBrazil.txt")
Crawler("https://www.milsons.com.br/",50,"Milsons.txt")
Crawler("https://www.playtech.com.br/",50,"PlayTech.txt")
Crawler("https://www.casasbahia.com.br/",50,"CasasBahia.txt")

"""
Crawler("http://www.novamusic.com.br/",1000,"Novamusic.txt")
Crawler("https://www.americanas.com.br/categoria/instrumentos-musicais",1000,"Americanas.txt")
Crawler("https://www.multisom.com.br/",1000,"Multisom.txt")
Crawler("https://www.mundomax.com.br/instrumentos-musicais",1000,"Mundomax.txt")
Crawler("https://www.madeinbrazil.com.br",1000,"MadeinBrazil.txt")
Crawler("https://www.milsons.com.br/",1000,"Milsons.txt")
Crawler("https://www.playtech.com.br/",1000,"PlayTech.txt")
Crawler("https://www.casasbahia.com.br/",1000,"CasasBahia.txt")
"""
