import session
import json
import time
import shutil
import requests
from pandas.io.json import json_normalize 
import pandas as pd
from operator import itemgetter

def import_summoners(data_path, verbose=0):
  s = session.init_session()

  # collect high ranks accounts
  queue = 'RANKED_SOLO_5x5'
  ranks = ['master', 'grandmaster', 'challenger']
  players = {}

  for rank in ranks:
    r = s.get(f'https://euw1.api.riotgames.com/lol/league/v4/{rank}leagues/by-queue/{queue}')
    players[rank] = r.json()['entries']

  with open(f'{data_path}/raw/player_without_id.txt', 'w') as outfile:
    json.dump(players, outfile)

  if verbose == 1:
    for rank in ranks:
      print(rank, len(players[rank]))

  # link to summoner name
  counter = 0
  nb_errors = 0
  for rank in ranks:
      for player in players[rank]:
          if ((counter % 100) == 0) & (verbose==1): print(counter)
          counter += 1
          name = player['summonerName']
          try:
            player['accountId'] = s.get(
              f'https://euw1.api.riotgames.com/lol/summoner/v4/summoners/by-name/{name}'
            ).json()['accountId']
          except KeyError:
            nb_errors += 1
            players[rank].remove(player)
          # for API rate
          time.sleep(1.5)
  if verbose==1: print(f'{nb_errors} errors')

  # export
  with open(f'{data_path}/raw/player_with_id.txt', 'w') as outfile:
    json.dump(players, outfile)

def import_matchs_id(data_path, verbose=0):
  s = session.init_session()

  # read players ID to use as keys for match history
  with open(f'{data_path}/raw/player_with_id.txt') as jsonfile:
      players = json.load(jsonfile)

  # we store match in set as their is a lot of duplicates
  all_match_id = set([])
  players_without_match = []

  counter = 0
  for rank, members in players.items():
      for member in members:
          if ((counter % 100) == 0) & (verbose): 
              print(counter)
              print(len(players_without_match))
          counter += 1
          try:
              encryptedAccountId = member['accountId']
              member_matches = s.get(
                  f'https://euw1.api.riotgames.com/lol/match/v4/matchlists/by-account/{encryptedAccountId}?queue=420'
                  ).json()['matches']
              # efficient parsing of json instead of yet another for loop
              member_matches = json_normalize(member_matches)['gameId'].to_list()
              all_match_id = all_match_id.union(set(member_matches))

          # no match in history            
          except KeyError:
              players_without_match.append(member)

          time.sleep(1.5)
  if verbose==1:
    print(f'{len(all_match_id)} matchs')
    print(f'{len(players_without_match)} players without match')

  # export
  pd.Series(
    list(all_match_id)
  ).to_csv(f'{data_path}/raw/match_id.csv', sep=';')


def id_to_champion(s):
  # get id to champion
  json_champ = s.get('http://ddragon.leagueoflegends.com/cdn/10.10.3216176/data/en_US/champion.json').json()['data']
  dict_champ = {}
  for name, champion in json_champ.items():
      dict_champ[int(champion['key'])] = name
  return dict_champ

def parse_match(matchId, s, dict_champ, verbose=0):
  """
  utilities to retrieve principal informations
  of a match from its ID
  """

  dict_id_to_color = {100: 'blue', 200: 'red'}
  # get match :)
  r = s.get(f'https://euw1.api.riotgames.com//lol/match/v4/matches/{matchId}')
  match_info = r.json()

  try:
      # check if soloqueue
      if match_info['queueId'] != 420:
          if verbose==1: print('wrong queue')
          return None


      # find winner
      winner = None
      for team in match_info['teams']:
          if team['win'] == 'Win':
              winner = dict_id_to_color[team['teamId']]
      duration = match_info['gameDuration']
      version = match_info['gameVersion']

      #find composition
      participants = []
      for participant in match_info['participants']:
          participants.append((
              dict_id_to_color[participant['teamId']],
              dict_champ[participant['championId']],
              # DUO, NONE, SOLO, DUO_CARRY, DUO_SUPPORT
              participant['timeline']['role'],
              # MIDDLE TOP JUNGLE BOTTOM
              participant['timeline']['lane'],
          ))
      participants = sorted(participants, key=itemgetter(0,3,2))
      # flatten and drop color (know with order)
      res = [matchId, duration, version, winner] + [_ for p in participants for _ in p[1:]]


      return res

  except KeyError:
      return None


def import_matchs(data_path, verbose):
  s = session.init_session()

  # structure
  colors = ['blue', 'red'] 
  colnames = ['champion', 'role', 'lane']
  numbers = range(5)

  # load match already stored if any
  try:
      matchs = pd.read_csv(f'{data_path}/matchs/matchs.csv', sep=';', index_col=0)
      # create temp save
      matchs.to_csv(f'{data_path}/matchs/matchs_tmp.csv', sep=';')
      if verbose == 1: print(len(matchs))
  except FileNotFoundError:
      matchs = pd.DataFrame(
          columns=['Id', 'time', 'version', 'winner'] + [
              f'{color}_{number}_{colname}' for color in colors for number in numbers for colname in colnames
          ]
      ) 
  # id to champion
  dict_champ = id_to_champion(s)

  # read match ids
  match_id = pd.read_csv(f'{data_path}/raw/match_id.csv', sep=';', index_col=0, header=None)
  match_id_to_query = set(match_id[1].to_list()) - set(matchs['Id'].to_list())
  
  # export every n iterations
  export_rate = 1000
  for counter, matchId in enumerate(match_id_to_query):
      res = parse_match(matchId, s, dict_champ, verbose)
      if res is not None:
          matchs.loc[matchId, :] = res
      if (counter%100 == 0) & (verbose==1): print(counter)
      time.sleep(1.5)
      if counter%export_rate == 0: matchs.to_csv(f'{data_path}/matchs/matchs.csv', sep=';')



def dl_and_save_img(data_path, champion_name, splash=True):
    if splash:
        adress = f"http://ddragon.leagueoflegends.com/cdn/img/champion/splash/{champion_name}_0.jpg"
    else:
        adress = f'http://ddragon.leagueoflegends.com/cdn/img/champion/loading/{champion_name}_0.jpg'

    r = requests.get(adress, stream=True)
    if r.status_code == 200:
        if splash:
            with open(f'{data_path}/images/splash/{champion_name}.png', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f) 
        else:
            with open(f'{data_path}/images/loading/{champion_name}.png', 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f) 


def import_splashs_art(data_path):
  # get all champions
  json_champ= requests.get('http://ddragon.leagueoflegends.com/cdn/10.10.3216176/data/en_US/champion.json').json()['data']
  all_champs = list(json_champ.keys())

  for champ in all_champs:
    dl_and_save_img(data_path, champ, splash=False)
  print('done')

