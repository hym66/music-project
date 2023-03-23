# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import selenium
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
import time

def print_hi(name):
    # 导入seleinum webdriver接口
    #options = webdriver.EdgeOptions()
    #options.add_argument('--headless')  # 无界面浏览
    browser = webdriver.Edge()
    browser.get('https://pixabay.com/zh/music/search/mood/')
    time.sleep(20)
    #browser.execute_script('window.scrollTo(0,document.body.scrollHeight)')
    listall=[]
    while True :
        list1 = list(set(browser.find_elements(By.LINK_TEXT,'下载')) - set(listall))
        #print(list1)
        if list1==[] :
            break
        listall.extend(list1)
        #print(len(list1))
        mouse=ActionChains(browser)
        for item in list1:
            mouse.click(item)
            mouse.perform()
            time.sleep(1)
            mouse.click(item)
            mouse.perform()
        #browser.execute_script('window.scrollBy(0,1000)')
    # 自动退出浏览器
    browser.quit()
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
