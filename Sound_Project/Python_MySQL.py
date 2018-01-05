import pymysql

# MySQL Connection 연결
db = pymysql.connect(host='localhost', port=3306, user='root', passwd='!a1s2d3f4', db='CCD', charset='utf8')

# Connection 으로부터 Cursor 생성
cursor = db.cursor()

# SQL문 실행
sql = "select * from SOUND"
cursor.execute(sql)

# 데이터 Fetch
data = cursor.fetchall()
print(data)

# Connection 닫기
db.close()
