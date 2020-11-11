

from credits import API_KEY
import requests
import webbrowser

def init_session():
    session = requests.Session()
    session.headers.update({'X-Riot-Token': API_KEY})
    test(session)
    return session

def test(session):
    # test 
    r = session.get('https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/Doublelift')
    try:
        if r.json()['id'] == '823HFGiucEaWhz8xBxoyBIlTRjwUWm74FCv9LET6VcWKVK0':
            print('API_KEY ok')
            return

    except:
        if r.json()['status']['status_code'] == 403:
            print('need to regenerate API key, and copy it to utils.credits.py')
            webbrowser.open('https://developer.riotgames.com/', 1)

