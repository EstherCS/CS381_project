import os
from bs4 import BeautifulSoup
import youtube_dl
import urllib.request
import requests
from gtts import gTTS
import sys
import urllib.parse

# keyword = '我們不一樣'
# keyword = urllib.parse.quote(keyword)
#
#
# url = "https://www.youtube.com/results?search_query="+keyword
# print(url)
def startPlayer():
    os.system("play.mp3")
def crawler(url):
    res = urllib.request.urlopen(url)
    html = res.read().decode('UTF-8')
    # print(html)
    soup = BeautifulSoup(html, "lxml")
    id_list = []
    for x in soup.select("li"):
        for id in x.select("a"):
            id_list.append(id['href'])
    x_list = []
    for x in id_list:
        if x.startswith("/watch?v="):
            x_list.append(x)
    last_id = x_list[0]
    last_id = last_id.replace('&list=','')
    #print(last_id)
    main = "https://www.youtube.com"
    video_list = main +last_id
    video = 'play.mp3'
    if os.path.isfile(video):
        os.remove(video)
    ydl_opts = {
        'format': 'bestaudio',
        'extractaudio': True,  # only keep the audio
        'audioformat': "mp3",  # convert to mp3
        'outtmpl': 'play.mp3',  # name the file the ID of the video
        'noplaylist': True,  # only download single song, not playlist
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_list])
    startPlayer()
    # command = "SDL_VIDEODRIVER=fbcon SDL_FBDEV=/dev/fb1 mplayer -vo sdl -framedrop play.mp3"
    # os.system(command)

# crawler(url)