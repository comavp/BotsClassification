import requests
from bs4 import BeautifulSoup as BS

HEADERS = 'Тут user-agent'
url = 'https://twitter.com/sessions'

screen_name_class_id = 'css-901oao css-16my406 r-poiln3 r-bcqeeo r-qvutc0'

session = requests.Session()
r = session.get(url, headers={
    'User-Agent': HEADERS
})

post_requests = session.post(url, {
     'username': 'Plankin Application',
     'password': '*(FHB(_*233hf89aphefiweufhOIHEIuhf9238h'
})

s = session.get('https://twitter.com/home', headers={
    'User-Agent': HEADERS,
})
soup = BS(s.content, "html.parser")
title = soup.find("span", class_='css-901oao css-16my406 r-1qd0xha r-ad9z0x r-bcqeeo r-qvutc0').text
print(title)