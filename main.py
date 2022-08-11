import base64
import io
import logging
import os
import random
from collections import defaultdict
from difflib import SequenceMatcher

import requests

# imports for local model. it's not needed if use torchserve
import torch
from flask import Flask, make_response, render_template, request
from flask_caching import Cache
from flask_socketio import SocketIO, join_room

from modeling.models.dalle.dalle import (
    MinDalle,
    get_tokenizer,
    post_process,
    prepare_tokens,
)

#######################################################################
#                         service endpoints                            #
#######################################################################

MAX_SCORE = 10
DEFAULT_TIMEOUT = 60

config = {
    "DEBUG": True,          # some Flask specific configs
    "CACHE_TYPE": "SimpleCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 3600,
    "SECRET_KEY": 'secret!',
}
app = Flask(__name__)
app.config.from_mapping(config)
logger = app.logger
logger.setLevel(logging.DEBUG)

socketio = SocketIO(app)
cache = Cache(app)

game_players_map = defaultdict(list)  # {game_id: list of players' user_name}
game_drawers_shuffled_map = defaultdict(list)  # {game_id: list of shuffled players' user_name}
round_id_map = defaultdict(int)  # {game_id: number indicating the game round}
game_sentences_map = defaultdict(list)  # {game_id: list of players' sentences}
game_creator_map = dict()
game_guesser_sentence_map = defaultdict(dict)  # {game_id: {guesser_name: sentence}}
game_leaderboard = defaultdict(dict)
round_leaderboard = defaultdict(dict)
game_timeout = defaultdict(lambda: DEFAULT_TIMEOUT)

game_rules = """
    1. We have N players join the game. N >= 2.
    2. A drawer will be randomly selected, the others become guessers.
    3. The drawer enters a sentence to start a round.
    4. AI will generate an image from the sentence and show it to all guessers.
    5. Each guesser enters a guessing sentence according to the image.
    6. Show a leaderboard with each guesser’s score(similarity to the correct sentence).
    7. Round ends. Start a new round and goto 2.
"""

@app.route('/')
def index():
    num_games=len(game_players_map)
    num_players = sum([len(players) for players in game_players_map.values()])
    return render_template('index.html', num_games=num_games, num_players=num_players)


@app.route('/welcome/<user_name>', methods=['GET'])
def welcome(user_name):
    logger.info(f"welcome {user_name}!")
    return render_template('welcome.html', user_name=user_name)


@app.route('/create_game/<game_id>/<user_name>', methods=['GET'])
def create_game(game_id, user_name):
    logger.info(f"User {user_name} created game {game_id}")
    game_creator_map[game_id] = user_name
    return render_template(
        'waiting_players.html',
        game_id=game_id,
        user_name=user_name,
        creator=user_name,
        game_rules=game_rules,
    )


@app.route('/join_game/<game_id>/<user_name>', methods=['GET'])
def join_game(game_id, user_name):
    if not game_creator_map.get(game_id, None):
        error_msg = f"game_id: {game_id} doesn't exist!"
        logger.error(error_msg)
        return make_response(error_msg, 200)
    creator = game_creator_map[game_id]
    return render_template(
        'waiting_players.html',
        game_id=game_id,
        user_name=user_name,
        creator=creator,
        game_rules=game_rules,
    )


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
    socketio.emit("start-game-event-response", json, callback=message_received, to=game_id)


@socketio.on('join-game-event')
def handle_join_game_event(json, methods=['GET', 'POST']):
    logger.info('received join-game-event: ' + str(json))
    game_id = json["game_id"]
    user_name = json["user_name"]
    
    # each user joins a room and related events will broadcast to the room
    join_room(game_id)
    logger.info(f"User {user_name} joined room {game_id}")
    
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
    socketio.emit(
        'join-game-event-response', 
        {"players": game_players_map[game_id]}, 
        callback=message_received,
        to=game_id,
    )


@socketio.on('join-room-event')
def handle_join_room_event(json):
    logger.info('handle join-room-event: ' + str(json))
    user_name = json["user_name"]
    game_id = json["game_id"]
    # each user joins a room and related events will broadcast to the room
    join_room(game_id)
    logger.info(f"User {user_name} joined room {game_id}")
    


@socketio.on('drawer-submit-event')
def handle_drawer_submit_event(json, methods=['GET', 'POST']):
    logger.info('received drawer-submit-event: ' + str(json))
    game_id = json["game_id"]
    game_sentences_map[json["game_id"]].clear()
    game_sentences_map[json["game_id"]].append(json["sentence"])
    json["masked_sentence"] = get_masked_sentence(json["sentence"])
    json["timeout"] = game_timeout[game_id]
    socketio.emit('drawer-submit-event-response', json, callback=message_received, to=game_id)

    # TODO: cache (draw's sentence, output image) for faster demo
    input_text = game_sentences_map[json["game_id"]][0] # assume first sentence is input
    model_name = json["model_name"]

    cache_key = f"{model_name}_{input_text}"
    images = cache.get(cache_key)
    
    if images is None:
        if model_name == "dalle_mini_local":
            images = predict_from_local_model(model_name, input_text)
        else:
            images = predict_from_torchserve(model_name, input_text)
    
    for image in images:
        img = get_encoded_img(img=image)
        socketio.emit('ai-returns-image-event', img, callback=message_received, to=game_id)
    
    # cache the result for fast demo or debugging
    if not cache.get(cache_key):
        cache.set(cache_key, [image])


@socketio.on('guesser-submit-event')
def handle_guesser_submit_event(json, methods=['GET', 'POST']):
    logger.info('received guesser-submit-event: ' + str(json))
    game_id = json["game_id"]
    user_name = json["user_name"]
    guess_sentence = json["sentence"]
    game_guesser_sentence_map[game_id][user_name] = guess_sentence
    correct_sentence = game_sentences_map[game_id][0]
    score = calculate_score(correct_sentence, guess_sentence)

    update_leaderboards(game_id, user_name, score)
    
    game_leaderboard_sorted = sorted(game_leaderboard[game_id].items(), key=lambda x: x[1], reverse=True)
    round_leaderboard_sorted = sorted(round_leaderboard[game_id].items(), key=lambda x: x[1], reverse=True)

    json["score"] = score
    json["game_leaderboard"] = game_leaderboard_sorted
    json["round_leaderboard"] = round_leaderboard_sorted
    
    json["is_guesser_win"] = abs(score - MAX_SCORE) <= 0.01

    socketio.emit('guesser-submit-event-response', json, callback=message_received, to=game_id)

    if json["is_guesser_win"]:
        # Guessers win as the guess score is close enough. start a new round
        start_new_round(game_id)


def start_new_round(game_id):
    drawer_name = get_drawer(game_id, round_id_map[game_id])
    drawer_score = get_drawer_score(game_id)
    update_leaderboards(game_id, drawer_name, drawer_score)
    json = {}
    json["drawer_name"] = drawer_name
    json["drawer_score"] = drawer_score
    json["correct_sentence"] = game_sentences_map[game_id][0]
    json["is_drawer_win"] = round_leaderboard[game_id][drawer_name] == MAX_SCORE
    game_leaderboard_sorted = sorted(game_leaderboard[game_id].items(), key=lambda x: x[1], reverse=True)
    round_leaderboard_sorted = sorted(round_leaderboard[game_id].items(), key=lambda x: x[1], reverse=True)
    json["game_leaderboard"] = game_leaderboard_sorted
    json["round_leaderboard"] = round_leaderboard_sorted
    reset_round_leaderboard(game_id)
    
    # clean up context for previous round
    game_guesser_sentence_map[game_id] = {}

    # setup context for the new round
    round_id_map[game_id] += 1
    json["round_id"] = round_id_map[game_id]
    json["new_drawer_name"] = select_drawer(game_id, round_id=round_id_map[game_id])
    socketio.emit("start-new-round-event", json, callback=message_received, to=game_id)


@socketio.on('timer-finish-event')
def handle_timer_finish_event(json, methods=['GET', 'POST']):
    logger.info('received timer-finish-event: ' + str(json))
    start_new_round(json["game_id"])


@socketio.on('update-time-out-event')
def handle_update_timeout(json, methods=['GET', 'POST']):
    logger.info('received update-time-out-event: ' + str(json))
    timeout = int(json["timeout"])
    assert timeout > 0
    game_id = json["game_id"]
    game_timeout[game_id] = timeout
    socketio.emit("update-time-out-event-response", json, callback=message_received, to=game_id)


#######################################################################
#                         helper functions                            #
#######################################################################


def predict_from_torchserve(model_name, input_text):
    logger.info(f'start tokenization, model: {model_name}, input_text: {input_text}')
    url = f"http://localhost:8080/predictions/{model_name}" # for dalle_image_mini
    # torchserve output is already post processed 
    images = [requests.post(url, data=input_text).content]
    return images


def predict_from_local_model(model_name, input_text):
    global model
    global tokenizer
    global device

    if model is None or tokenizer is None:
        model, tokenizer = build_model_and_tokenizer()

    logger.info(f'start tokenization, model: {model_name}, input_text: {input_text}')
    tokens = prepare_tokens(tokenizer, input_text, device=device)
    logger.info(f'start prediction, tokens: {tokens}')
    images = model(
        tokens,
        grid_size=1,
        temperature=1,
        top_k=32,
        supercondition_factor=16.0,
        progressive_outputs=True,
        is_seamless= False,
        is_verbose= True,
    )
    return post_process(images)


def calculate_score(correct_sentence, guess_sentence):
    # TODO: call text-similarity model to calculate the score
    score = SequenceMatcher(None, correct_sentence, guess_sentence).ratio()
    score = round(score * MAX_SCORE, 1)
    return  score


def get_drawer_score(game_id):
    """
    If no guessers get MAX_SCORE, the drawer will get MAX_SCORE. Otherwise, drawer gets 0
    It incentivizes the drawer to enter a harder sentence to fight against the guessers
    """
    for score in round_leaderboard[game_id].values():
        if score >= MAX_SCORE:
            return 0
    return MAX_SCORE


def reset_leaderboards(game_id):
    reset_round_leaderboard(game_id)
    
    for player in game_players_map[game_id]:
        game_leaderboard[game_id][player] = 0


def reset_round_leaderboard(game_id):    
    for player in game_players_map[game_id]:
        round_leaderboard[game_id][player] = 0


def update_leaderboards(game_id, user_name, score):
    # only update leaderboard for the higher score or the first score
    if score >= round_leaderboard[game_id][user_name]:
        game_leaderboard[game_id][user_name] += score - round_leaderboard[game_id][user_name]
        round_leaderboard[game_id][user_name] = score


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


def get_masked_sentence(sentence):
    return "".join([" " if c == " " else "*" for c in sentence])

#######################################################################
#                   Build tokenizer and model                         #
#######################################################################

model = None
tokenizer = None
device = "cuda:1"


def build_model_and_tokenizer():
    logger.info(f'start build_model_and_tokenizer...')
        
    global model
    global tokenizer
    global device
    
    is_mega = False
    root_dir = "modeling/pretrained"

    torch.manual_seed(42)
    tokenizer = get_tokenizer(os.path.join(root_dir, f"dalle_{'mega' if is_mega else 'mini'}"))
    model = MinDalle(is_mega=is_mega, root_dir="modeling/pretrained", is_reusable=True, device=device)
    return model, tokenizer


if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=9000)
