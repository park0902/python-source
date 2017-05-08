def password_check(pwd):
    """ checking password   
    Args:     
        pwd(str) : password check
    
    Return:  
        True or False (boolean) : the result of checking """

    if len(pwd) < 6 or len(pwd) > 12:
        print(pwd, '의 길이가 적당하지 않습니다')
        return False

    for c in pwd:
        if not c.isnumeric() and not c.isalpha():
            print(pwd, '숫자와 문자로 되어있지 않습니다')
            return False

    print(pwd, '는 비밀번호로 적당합니다')
    return True

print(password_check('%%%@'))