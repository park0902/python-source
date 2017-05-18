# import pandas as pd
#
#
# emp = pd.read_csv("d:\data\ex.txt", names=['a','b','c','d','e','f','g'])
# print(emp.plot())

#
# import pandas as pd
#
# emp = pd.DataFrame.from_csv("d:\data\emp.csv")
# result = emp[['ename', 'sal']][emp['sal'] >= 3000]
#             #열 조건                #행 조건
# print(result)


import pandas as pd

emp = pd.DataFrame.from_csv("d:\data\emp.csv")
result = emp[['ename', 'sal']][emp['sal'] >= 3000].sort_values('sal',ascending=True)
print(result)


# import urllib.request  # 웹브라우저에서 html 문서를 얻어오기위해 통신하는 모듈
# from  bs4 import BeautifulSoup  # html 문서 검색 모듈
# import os
# import re
#
# def get_save_path():
#     save_path = input("Enter the file name and file location : ")
#     save_path = save_path.replace("\\", "/")
#
#     if not os.path.isdir(os.path.split(save_path)[0]):
#         os.mkdir(os.path.split(save_path)[0])   # 폴더가 없으면 만드는 작업
#     return save_path
#
# def fetch_list_url():
#     params = []
#     for j in range(1, 30):
#
#         list_url = "http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_lis.jsp"
#
#         request_header = urllib.parse.urlencode({"page": j})
#         # print (request_header) # 결과 page=1, page=2 ..
#
#         request_header = request_header.encode("utf-8")
#         # print (request_header) # b'page=29'
#
#         url = urllib.request.Request(list_url, request_header)
#         # print (url) # <urllib.request.Request object at 0x00000000021FA2E8>
#
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#
#         soup = BeautifulSoup(res, "html.parser")
#         soup2 = soup.find_all("li", class_="pclist_list_tit2")
#         for soup3 in soup2:
#             soup4 = soup3.find("a")["href"]
#             params.append(re.search("[0-9]{14}", soup4).group())
#
#
#     return params
#
#
# def fetch_list_url2():
#
#     params2 = fetch_list_url()
#     f = open(get_save_path(), 'w', encoding='utf-8')
#     for i in params2:
#
#         detail_url = "http://eungdapso.seoul.go.kr/Shr/Shr01/Shr01_vie.jsp"
#
#         request_header = urllib.parse.urlencode({"RCEPT_NO": str(i) })
#         request_header = request_header.encode("utf-8")
#
#         url = urllib.request.Request(detail_url, request_header)
#         res = urllib.request.urlopen(url).read().decode("utf-8")
#         soup = BeautifulSoup(res, "html.parser")
#         soup2 = soup.find("div", class_="form_table")
#
#         tables = soup2.find_all("table")
#         table0 = tables[0].find_all("td")
#         table1 = tables[1].find("div",class_="table_inner_desc")
#         table2 = tables[2].find("div",class_="table_inner_desc")
#
#         date = table0[1].get_text()
#         title = table0[0].get_text()
#         question = table1.get_text(strip=True)
#         answer = table2.get_text(strip=True)
#         # print(answer)
#         f.write('=='*30 + '\n')
#         f.write(title + '\n')
#         f.write(date + '\n')
#         f.write(question + '\n')
#         f.write(answer + '\n')
#         f.write('=='*30 + '\n')
#
#     f.close()
#
# fetch_list_url2()

