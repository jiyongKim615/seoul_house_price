# 검색어 입력
page = 1
page2 = 100
s_date = '2019.02.01'
e_date = '2019.02.31'
for search in ['부동산', '주택', '아파트', '청약', '재건축', '분양', '재개발', '집값']:
    # search = input("검색할 키워드를 입력해주세요:")
    # #검색 시작할 페이지 입력
    # page = int(input("\n크롤링할 시작 페이지를 입력해주세요. ex)1(숫자만입력):")) # ex)1 =1페이지,2=2페이지...
    # print("\n크롤링할 시작 페이지: ",page,"페이지")
    # #검색 종료할 페이지 입력
    # page2 = int(input("\n크롤링할 종료 페이지를 입력해주세요. ex)1(숫자만입력):")) # ex)1 =1페이지,2=2페이지...
    # print("\n크롤링할 종료 페이지: ",page2,"페이지")

    # s_date = input("시작 날짜(YYYY.MM.DD): ")
    # e_date = input("종료 날짜(YYYY.MM.DD): ")
    # naver url 생성

    # for search in ['부동산', '주택', '아파트', '집값', '청약', '분양', '재개발', '재건축']

    url = makeUrl(search, page, page2, s_date, e_date)

    # 뉴스 크롤러 실행
    news_titles = []
    news_url = []
    news_contents = []
    news_dates = []
    for i in url:
        url = articles_crawler(url)
        news_url.append(url)


    # 제목, 링크, 내용 1차원 리스트로 꺼내는 함수 생성
    def makeList(newlist, content):
        for i in content:
            for j in i:
                newlist.append(j)
        return newlist


    # 제목, 링크, 내용 담을 리스트 생성
    news_url_1 = []

    # 1차원 리스트로 만들기(내용 제외)
    makeList(news_url_1, news_url)

    # NAVER 뉴스만 남기기
    final_urls = []
    for i in tqdm(range(len(news_url_1))):
        if "news.naver.com" in news_url_1[i]:
            final_urls.append(news_url_1[i])
        else:
            pass

    # 뉴스 내용 크롤링

    for i in tqdm(final_urls):
        # 각 기사 html get하기
        news = requests.get(i, headers=headers)
        news_html = BeautifulSoup(news.text, "html.parser")

        # 뉴스 제목 가져오기
        title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        if title == None:
            title = news_html.select_one("#content > div.end_ct > div > h2")

        # 뉴스 본문 가져오기
        content = news_html.select("div#dic_area")
        if content == []:
            content = news_html.select("#articeBody")

        # 기사 텍스트만 가져오기
        # list합치기
        content = ''.join(str(content))

        # html태그제거 및 텍스트 다듬기
        pattern1 = '<[^>]*>'
        title = re.sub(pattern=pattern1, repl='', string=str(title))
        content = re.sub(pattern=pattern1, repl='', string=content)
        pattern2 = """[\n\n\n\n\n// flash 오류를 우회하기 위한 함수 추가\nfunction _flash_removeCallback() {}"""
        content = content.replace(pattern2, '')

        news_titles.append(title)
        news_contents.append(content)

        try:
            html_date = news_html.select_one(
                "div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > span")
            news_date = html_date.attrs['data-date-time']
        except AttributeError:
            news_date = news_html.select_one("#content > div.end_ct > div > div.article_info > span > em")
            news_date = re.sub(pattern=pattern1, repl='', string=str(news_date))
        # 날짜 가져오기
        news_dates.append(news_date)

    print("검색된 기사 갯수: 총 ", (page2 + 1 - page) * 10, '개')
    print("\n[뉴스 제목]")
    print(news_titles)
    print("\n[뉴스 링크]")
    print(final_urls)
    print("\n[뉴스 내용]")
    print(news_contents)

    print('news_title: ', len(news_titles))
    print('news_url: ', len(final_urls))
    print('news_contents: ', len(news_contents))
    print('news_dates: ', len(news_dates))

    ###데이터 프레임으로 만들기###
    import pandas as pd

    # 데이터 프레임 만들기
    news_df = pd.DataFrame({'date': news_dates, 'title': news_titles, 'link': final_urls, 'content': news_contents})

    # 중복 행 지우기
    news_df = news_df.drop_duplicates(keep='first', ignore_index=True)
    print("중복 제거 후 행 개수: ", len(news_df))

    # 데이터 프레임 저장
    now = datetime.datetime.now()
    news_df['date'] = s_date
    news_df.to_csv('{}_{}.csv'.format(search, now.strftime('%Y%m%d_%H시%M분%S초')), encoding='utf-8-sig', index=False)