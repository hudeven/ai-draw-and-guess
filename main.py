import base64
import io
import logging
import requests
import os

from PIL import Image
from collections import defaultdict
from difflib import SequenceMatcher
from flask import Flask, render_template, request
from flask_socketio import SocketIO
import random


#######################################################################
#                   Build tokenizer and model                         #
#######################################################################


# import torch
# from modeling.models.dalle.dalle import MinDalle, get_tokenizer, prepare_tokens
# from PIL import Image


# root_dir = "modeling/pretrained"
# is_mega = False

# torch.manual_seed(42)
# tokenizer = get_tokenizer(os.path.join(root_dir, f"dalle_{'mega' if is_mega else 'mini'}"))
# model = MinDalle(is_mega=False, root_dir="modeling/pretrained")


#######################################################################
#                         service endpoints                            #
#######################################################################


app = Flask(__name__)
logger = app.logger
logger.setLevel(logging.DEBUG)
socketio = SocketIO(app)

game_players_map = defaultdict(list)  # {game_id: list of players' user_name}
game_drawers_shuffled_map = defaultdict(list)  # {game_id: list of shuffled players' user_name}
round_id_map = defaultdict(int)  # {game_id: number indicating the game round}
game_sentences_map = defaultdict(list)  # {game_id: list of players' sentences}
game_creator_map = dict()
game_guesser_sentence_map = defaultdict(dict)  # {game_id: {guesser_name: sentence}}
game_leaderboard = defaultdict(dict)
round_leaderboard = defaultdict(dict)


@app.route('/')
def index():
    logger.info("hello!")
    return render_template('index.html')


@app.route('/welcome/<user_name>', methods=['GET'])
def welcome(user_name):
    logger.info(f"welcome {user_name}!")
    return render_template('welcome.html', user_name=user_name)


@app.route('/create_game/<game_id>/<user_name>', methods=['GET'])
def create_game(game_id, user_name):
    logger.info(f"User {user_name} created game {game_id}")
    game_creator_map[game_id] = user_name
    return render_template('waiting_players.html', game_id=game_id, user_name=user_name, creator=user_name)


@app.route('/join_game/<game_id>/<user_name>', methods=['GET'])
def join_game(game_id, user_name):
    if not game_creator_map.get(game_id, None):
        logger.error(f"Invalid game_id: {game_id}")
        return
    creator = game_creator_map[game_id]
    return render_template('waiting_players.html', game_id=game_id, user_name=user_name, creator=creator)


@app.route('/game_loop/<game_id>/<user_name>', methods=['GET', 'POST'])
def game_loop(game_id, user_name):
    logger.info(f"game: {game_id}, user {user_name} enters game_loop")

    round_id = 0
    drawer_name = select_drawer(game_id, round_id)
    logger.info(f"drawer: {drawer_name} is selected for this round!")
    return render_template(
        'game_loop.html',
        game_id=game_id,
        user_name=user_name,
        drawer_name=drawer_name,
        round_id=round_id,
    )


#######################################################################
#                  Socket.IO event handlers                            #
#######################################################################


@socketio.on('start-game-event')
def handle_start_game_event(json, methods=['GET', 'POST']):
    logger.info('received start-game-event: ' + str(json))
    game_id = json["game_id"]
    round_id_map[game_id] = 0
    reset_leaderboards(game_id)
    json["round_id"] = 0
    socketio.emit("start-game-event-response", json, callback=message_received)


@socketio.on('join-game-event')
def handle_join_game_event(json, methods=['GET', 'POST']):
    logger.info('received join-game-event: ' + str(json))
    game_id = json["game_id"]
    user_name = json["user_name"]
    if not game_id:
        logger.error(f"bad game id: {game_id}")
        return
    if not user_name:
        logger.error(f"bad user_name: {user_name}")
        return

    if user_name not in game_players_map[game_id]:
        game_players_map[game_id].append(user_name)
    else:
        logger.error(f"Duplicated user_name: {user_name} !")
    logger.info(f"return players: {game_players_map[game_id]}")
    socketio.emit('join-game-event-response', {"players": game_players_map[game_id]}, callback=message_received)


@socketio.on('drawer-submit-event')
def handle_drawer_submit_event(json, methods=['GET', 'POST']):
    logger.info('received drawer-submit-event: ' + str(json))
    game_sentences_map[json["game_id"]].clear()
    game_sentences_map[json["game_id"]].append(json["sentence"])
    socketio.emit('drawer-submit-event-response', json, callback=message_received)

    # TODO: cache (draw's sentence, output image) for faster demo
    input_text = game_sentences_map[json["game_id"]][0] # assume first sentence is input
    model_name = json["model_name"]

    url = f"http://localhost:8080/predictions/{model_name}" # for dalle_image_mini
    response = requests.post(url, data=input_text)

    img = get_encoded_img(response.content)
    socketio.emit('ai-returns-image-event', img, callback=message_received)


@socketio.on('guesser-submit-event')
def handle_guesser_submit_event(json, methods=['GET', 'POST']):
    logger.info('received guesser-submit-event: ' + str(json))
    game_id = json["game_id"]
    user_name = json["user_name"]
    guess_sentence = json["sentence"]
    game_guesser_sentence_map[game_id][user_name] = guess_sentence
    correct_sentence = game_sentences_map[game_id][0]

    # TODO: call text-similarity model to calculate the score
    score = SequenceMatcher(None, correct_sentence, guess_sentence).ratio() * 10
    ######################################################################

    update_leaderboards(game_id, user_name, score)

    round_end = is_round_end(game_id)
    json["is_round_end"] = round_end
    if round_end:
        # at the end of round, add score for the drawer
        drawer_name = get_drawer(game_id, round_id_map[game_id])
        drawer_score = get_drawer_score(game_id)
        update_leaderboards(game_id, drawer_name, drawer_score)
        json["round_id"] = round_id_map[game_id]
        json["drawer_name"] = drawer_name
        json["drawer_score"] = drawer_score
        json["is_drawer_win"] = game_leaderboard[game_id][drawer_name] == 10

    game_leaderboard_sorted = sorted(game_leaderboard[game_id].items(), key=lambda x: x[1], reverse=True)
    round_leaderboard_sorted = sorted(round_leaderboard[game_id].items(), key=lambda x: x[1], reverse=True)

    json["score"] = score
    json["game_leaderboard"] = game_leaderboard_sorted
    json["round_leaderboard"] = round_leaderboard_sorted

    socketio.emit('guesser-submit-event-response', json, callback=message_received)

    if round_end:
        # clean up context for previous round
        game_guesser_sentence_map[game_id] = {}

        # setup context for the new round
        round_id_map[game_id] += 1
        drawer_name = select_drawer(game_id, round_id=round_id_map[game_id])
        payload = {"game_id": game_id, "round_id": round_id_map[game_id], "drawer_name": drawer_name}
        socketio.emit("start-new-round-event", payload, callback=message_received)


#######################################################################
#                         helper functions                            #
#######################################################################


def get_drawer_score(game_id):
    """
    If no guessers get 10, the drawer will get 10. Otherwise, drawer gets 0
    It incentivizes the drawer to enter a harder sentence to fight against the guessers
    """
    for user_name, score in round_leaderboard[game_id].items():
        if score >= 10:
            return 0
    return 10


def reset_leaderboards(game_id):
    players = game_players_map[game_id]
    for player in players:
        game_leaderboard[game_id][player] = 0
        round_leaderboard[game_id][player] = 0


def update_leaderboards(game_id, user_name, score):
    round_leaderboard[game_id][user_name] = score
    game_leaderboard[game_id][user_name] += score


def select_drawer(game_id, round_id):
    round_id = int(round_id)
    if round_id == 0:
        random.seed(0)
        players = game_players_map[game_id]
        game_drawers_shuffled_map[game_id] = random.sample(players, len(players))

    drawer_name = get_drawer(game_id, round_id)
    logger.info(f"drawer: {drawer_name} is selected for this round!")
    return drawer_name


def get_drawer(game_id, round_id):
    drawer_id = round_id % len(game_drawers_shuffled_map[game_id])
    return game_drawers_shuffled_map[game_id][drawer_id]


def message_received(methods=['GET', 'POST']):
    logger.info('message was received!!!')


def is_round_end(game_id):
    num_guesser_sentences = len(game_guesser_sentence_map[game_id])
    num_guesser = len(game_players_map[game_id]) - 1
    return num_guesser_sentences == num_guesser

def get_encoded_img(img):
    encoded_img = base64.encodebytes(img).decode('ascii')
    return encoded_img


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=9000)
