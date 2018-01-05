import pymysql

# MySQL Connection 연결
db = pymysql.connect(host='localhost', port=3306, user='root', passwd='!a1s2d3f4', db='CCD', charset='utf8')

# Connection 으로부터 Cursor 생성
cursor = db.cursor(pymysql.cursors.DictCursor)

# SQL문 실행
sql = "select * from SOUND where SOUND_KEY = %s"
cursor.execute(sql, ('abc-01'))

# 데이터 Fetch
datas = cursor.fetchall()

for data in datas:
    print(data)
    print(data['NO'], data['AVG'], data['SOUND_KEY'])

# Connection 닫기
db.close()
