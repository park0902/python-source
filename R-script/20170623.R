# 월급이 1200이상이고 직업이 SALESMAN 인 사원들의 이름과 월급과 직업을 출력하는데
# 월급이 높은 사원부터 출력

emp <- read.csv('emp.csv', header = TRUE)

emp2 <- emp[emp$sal >= 1200 & emp$job == 'SALESMAN', c('ename', 'sal', 'job')]

emp2[order(emp2$sal, decreasing = T), c("ename", "sal", "job")]



# 이름의 세번째 철자가 M 인 사원들의 이름과 월급과 직업을 출력
emp <- read.csv('emp.csv', header = TRUE)

emp[ grep("^..M", emp$ename) , c('ename', 'sal', 'job')]
# emp[ grep("^.{2}M.*$", emp$ename) , c('ename', 'sal', 'job')]



# 직업이 SALESMAN 인 사원들의 부서번호를 출력하는데 중복제거해서 출력
emp <- read.csv('emp.csv', header = TRUE)

unique(emp[emp$job == 'SALESMAN', c('deptno')])



# doBy 사용 예
install.packages("doBy")
library(doBy)

emp <- read.csv('emp.csv', header = TRUE)
orderBy(~sal, emp[ , c("ename", "sal")])   # asc
orderBy(~-sal, emp[ , c("ename", "sal")])  # desc



# 직업이 ANALYST 가 아닌 사원들의 이름과 월급과 직업을 출력하는데 월급이 높은 사원부터 출력
install.packages("doBy")
library(doBy)

emp <- read.csv('emp.csv', header = TRUE)

orderBy(~-sal, emp[emp$job != "ANALYST", c("ename", "sal", "job")])



# 일요일에 발생하는 범죄유형,범죄건수를 출력하는데 범죄건수가 높은것부터 출력
install.packages("doBy")
library(doBy)

crime_day <- read.csv('crime_day.csv', header = TRUE)

orderBy(~-CNT, crime_day[crime_day$DAY == "SUN", c("C_T", "CNT")])


# 현재 사용하고 있는 변수의 목록 확인방법
ls()


# 변수를 지우는 방법
# rm(변수명)



# 살인이 일어나는 장소와 건수 출력하는데 살인이 일어나는 건수가 높은 순으로 출력
install.packages("doBy")
library(doBy)

crime_loc <- read.csv('crime_loc.csv', header = TRUE)

orderBy(~-건수, crime_loc[crime_loc$범죄 == "살인", c("장소", "건수")])



# 강도가 가장 많이 발생하는 장소는 어디인지 출력
install.packages("doBy")
library(doBy)

crime_loc <- read.csv('crime_loc.csv', header = TRUE)

loc_max <- orderBy(~-건수, crime_loc[crime_loc$범죄 == "강도", c("장소", "건수")])

loc_max[1,1]



# 금요일에 가장 많이 발생하는 범죄가 무엇인지 출력
install.packages("doBy")
library(doBy)

crime_day <- read.csv('crime_day.csv', header = TRUE)
head(crime_day)

day_max <- orderBy(~-CNT, crime_day[crime_day$DAY == "FRI", c("C_T", "CNT")])

day_max[1,1]



# 이름과 월급을 출력하는데 소문자로 출력
library(data.table)
emp <- read.csv('emp.csv', header = TRUE)

data.table(이름 = tolower(emp$ename), 직업 = tolower(emp$job))



# 이름을 입력하면 해당 사원의 이름과 월급이 출력되는 R 코드를 작성하는데 이름을 소문자로 입력해도 출력
emp <- read.csv('emp.csv', header = TRUE)

find_sal <- function() {
  
  response <- toupper(readline(prompt = "이름이 뭐에요? "))
  x <- emp[emp$ename == response, c("ename", "sal")]
  print(x)
                }

find_sal()



# 요일을 물어보게하고 요일을 입력하면 해당 요일의 가장 많이 발생하는 범죄유형 출력
install.packages("doBy")
library(doBy)

crime_day <- read.csv('crime_day.csv', header = TRUE)

find_crime <- function() {
  
  response <- toupper(readline(prompt = "요일을 입력하세요! "))
  x <- orderBy(~-CNT,crime_day[crime_day$DAY == response, c("C_T", "CNT")])
  print(x[1,1])
  
}

find_crime()



# 이름을 출력하고  그 이름 옆에 이름의 첫번째 철자부터 세번째 철자 출력
library(data.table)
emp <- read.csv('emp.csv', header = TRUE)

data.table(emp$ename, substr(emp$ename,1,3))



#이름의 두번째 철자가 M인 사원들의 이름과 월급을 출력하는데 substr 함수를 사용해서 출력
emp <- read.csv('emp.csv', header = TRUE)

emp[substr(emp$ename,2,2) == "M", c("ename", "sal")]



# 이름, 월급을 출력하는데 월급을 출력할때 0으로 출력
emp <- read.csv('emp.csv', header = TRUE)

data.table(emp$ename, gsub('0', '*', emp$sal))



# 이름과 월급을 출력하는데 월급을 출력할때에 숫자 0,1,2, 를 *로 출력
emp <- read.csv('emp.csv', header = TRUE)

data.table(emp$ename, gsub('[0-2]', "*", emp$sal))



# 6의 9승 출력
6^9



# 10을 3으로 나눈 나머지 값 출력
10%%3



# 이름과 연봉을 출력하는데 연봉은 월급의 12를 곱해서 출력하고 컬럼명이 한글로 연봉으로 출력
emp <- read.csv('emp.csv', header = TRUE)

data.table(ename = emp$ename, '연봉' = emp$sal*12)



# 위의 결과를 다시 출력하는데 round 함수를 써서 백의 자리에서 반올림 해서 출력
emp <- read.csv('emp.csv', header = TRUE)

data.table(ename = emp$ename, '연봉' = round(emp$sal*12,-3))



# 문제37번의 결과를 다시 출력하는데 백자리 이후를 다 버리고 출력
emp <- read.csv('emp.csv', header = TRUE)

data.table(ename = emp$ename, '연봉' = trunc(emp$sal*12, -3))



# 오늘 날짜 출력
Sys.Date()



# 이름, 입사한 날짜부터 오늘까지 총 몇일 근무했는지 출력
emp <- read.csv('emp.csv', header = TRUE)

data.frame(emp$ename, Sys.Date()- as.Date(emp$hiredate))



# 이름, 입사한 날짜부터 오늘까지 총 몇달 근무했는지 출력
emp <- read.csv('emp.csv', header = TRUE)



# 오늘날짜의 달의 마지막 날짜 출력
install.packages("lubridate")
library(lubridate)

emp <- read.csv('emp.csv', header = TRUE)

last_day <- function(x) {
  
  ceiling_date(x, 'month') - days(1)
      
  }

last_day(Sys.Date())



# last_day 함수처럼 first_day 함수 생성
install.packages("lubridate")
library(lubridate)

emp <- read.csv('emp.csv', header = TRUE)

first_day <- function(x) {
  
  ceiling_date(x, 'months') - months(1)

  }

first_day(Sys.Date())



# select next_day(sysdate, '월요일') from dual; SQL 결과를 R로 구현!
next_day <- function(x, day){
  for (y in 1:7) {
    check_date = as.Date( x + days(y) )
    if (format(check_date, '%A') == day){
      print(check_date)
    }
  }
}

next_day(Sys.Date(), "금요일")



# 이름, 입사한 요일 출력
emp <- read.csv('emp.csv', header = TRUE)

data.table(emp$ename, format(as.Date(emp$hiredate), '%A'))



# 내가 무슨 요일에 태어났는지 출력
format(as.Date('1991/09/02'), '%A')



# 오늘부터 100달 뒤에 돌아오는 날짜의 요일 출력
format(Sys.Date() + months(100), '%A')



# 이름, 월급, 등급을 출력하는데 등급이 월급이 1500 이상이면 A등급, 아니면 B 등급으로 출력
emp <- read.csv('emp.csv', header = TRUE)

data.table(emp$ename, emp$sal, ifelse(emp$sal >= 1500, 'A', 'B'))



# 이름, 월급, 등급을 출력하는데 등급이 월급이 3000이상이면 A, 1500 이상이고 3000보다 작으면 B, 나머지 사원들은 C 를 출력
emp <- read.csv('emp.csv', header = TRUE)

data.table(emp$ename, emp$sal, ifelse(emp$sal >= 3000, 'A', 
                               ifelse(emp$sal >= 1500 & emp$sal <= 3000, 'B', 'C')))




format(Sys.Date(), '%A')  # 요일
format(Sys.Date(), '%Y')  # 년도
format(Sys.Date(), '%d')  # 일
format(Sys.Date(), '%b')  # 달


# 이름, 월급, 보너스를 출력하는데 1980년도에 입사했으면 보너스를 A를 출력, 81년도 B, 82년도 C, 나머지는 D 출력
emp <- read.csv('emp.csv', header = TRUE)

data.table(ename=emp$ename, hiredate=emp$hiredate,
           bouns=ifelse(format(as.Date(emp$hiredate), '%Y') == '1980', 'A',
           ifelse(format(as.Date(emp$hiredate), '%Y') == '1981', 'B', 
           ifelse(format(as.Date(emp$hiredate), '%Y') == '1982', 'C', 'D'))))



# is.na 함수를 이용해서 커미션이 NA인 사원들의 이름과 커미션 출력
emp <- read.csv('emp.csv', header = TRUE)

emp[is.na(emp$comm), c('ename', 'comm')]



# 이름과 커미션을 출력하는데 커미션이 NA인 사원들은  no comm 으로 출력
emp <- read.csv('emp.csv', header = TRUE)

data.table(ename = emp$ename,
           comm = ifelse(is.na(emp$comm), 'no comm', emp$comm))



# 최대 월급 출력
emp <- read.csv('emp.csv', header = TRUE)

max(emp$sal)



# 직업, 직업별 최대 월급 출력
emp <- read.csv('emp.csv', header = TRUE)

aggregate(sal~job, emp, max)



# 부서번호, 부서번호별 최대 월급을 출력하는데 부서번호별 최대월급을 높은 순으로 출력
emp <- read.csv('emp.csv', header = TRUE)

x <- aggregate(sal~deptno, emp, max)

names(x) <- c('deptno', 'maxsal')

orderBy(~-maxsal, x)



# 직업, 직업별 인원수 출력
emp <- read.csv('emp.csv', header = TRUE)

x <- aggregate(empno~job, emp, length)  # 세로
#x <- table(emp$job)   # 가로

names(x) <- c('job', 'cnt')

orderBy(~-cnt, x)
barplot(x, col=rainbow(5))



# 막대그래프 그리기
x <- table(emp$job)
barplot(x, col=rainbow(5), main="직업별 인원수", density=50)




# 입사한 년도(4자리), 입사한 년도별 토탈월급을 출력하는데 그 결과를 막대 그래프 시각화
emp <- read.csv('emp.csv', header = TRUE)

x <- aggregate(sal~format(as.Date(emp$hiredate), '%Y'), emp, sum)

names(x) <- c('hiredate', 'sumsal')

barplot(x$sumsal, names.arg=x$hiredate, col=rainbow(4), main = "연도별 토탈 월급" ,density=50)





