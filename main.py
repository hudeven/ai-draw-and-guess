import base64
import io
import logging
import requests
import os

from collections import defaultdict
from difflib import SequenceMatcher
from flask import Flask, render_template, request
from flask_socketio import SocketIO, join_room
import random


# imports for local model. it's not needed if use torchserve
import torch
from modeling.models.dalle.dalle import MinDalle, get_tokenizer, prepare_tokens, post_process



#######################################################################
#                         service endpoints                            #
#######################################################################


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
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
        logger.error(f"Invalid game_id: {game_id}")
        return
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
    socketio.emit('drawer-submit-event-response', json, callback=message_received, to=game_id)

    # TODO: cache (draw's sentence, output image) for faster demo
    input_text = game_sentences_map[json["game_id"]][0] # assume first sentence is input
    model_name = json["model_name"]

    if model_name == "dalle_mini_local":
        global model
        global tokenizer
        
        if model is None or tokenizer is None:
            model, tokenizer = build_model_and_tokenizer()
        
        logger.info(f'start tokenization, model: {model_name}')
        tokens = prepare_tokens(tokenizer, input_text)
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
        images = post_process(images)
    else:
        url = f"http://localhost:8080/predictions/{model_name}" # for dalle_image_mini
        # torchserve output is already post processed 
        images = [requests.post(url, data=input_text).content]
    
    for image in images:
        img = get_encoded_img(img=image)
        socketio.emit('ai-returns-image-event', img, callback=message_received, to=game_id)



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

    socketio.emit('guesser-submit-event-response', json, callback=message_received, to=game_id)

    if round_end:
        # clean up context for previous round
        game_guesser_sentence_map[game_id] = {}

        # setup context for the new round
        round_id_map[game_id] += 1
        drawer_name = select_drawer(game_id, round_id=round_id_map[game_id])
        payload = {"game_id": game_id, "round_id": round_id_map[game_id], "drawer_name": drawer_name}
        socketio.emit("start-new-round-event", payload, callback=message_received, to=game_id)


#######################################################################
#                         helper functions                            #
#######################################################################


def calculate_score(correct_sentence, guess_sentence):
    # TODO: call text-similarity model to calculate the score
    score = SequenceMatcher(None, correct_sentence, guess_sentence).ratio()
    score = round(score * 10, 1)
    return  score


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


#######################################################################
#                   Build tokenizer and model                         #
#######################################################################

model = None
tokenizer = None


def build_model_and_tokenizer():
    logger.info(f'start build_model_and_tokenizer...')
        
    global model
    global tokenizer
    
    is_mega = False
    root_dir = "modeling/pretrained"

    torch.manual_seed(42)
    tokenizer = get_tokenizer(os.path.join(root_dir, f"dalle_{'mega' if is_mega else 'mini'}"))
    model = MinDalle(is_mega=is_mega, root_dir="modeling/pretrained", is_reusable=True)
    return model, tokenizer


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=9000)
